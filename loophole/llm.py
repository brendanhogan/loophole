from __future__ import annotations

import atexit
import json
import select
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Any

import anthropic


@dataclass(slots=True)
class CodexAppServerConfig:
    command: str = "codex"
    model: str = "gpt-5.3-codex"
    sandbox: str = "read-only"
    approval_policy: str = "never"
    timeout_seconds: float = 180.0
    ephemeral: bool = True
    developer_instructions: str = (
        "You are a pure text completion engine used by a Python program. "
        "Never run tools, shell commands, or file operations. "
        "Do not mention internal policies or startup instructions. "
        "Respond only to the task content."
    )


class CodexAppServerClient:
    def __init__(self, config: CodexAppServerConfig):
        if shutil.which(config.command) is None:
            raise RuntimeError(
                f"'{config.command}' was not found in PATH. Install Codex CLI and run `codex login`."
            )

        self.config = config
        self._id_counter = count(1)
        self._stderr_tail: deque[str] = deque(maxlen=100)

        self._proc = subprocess.Popen(
            [config.command, "app-server", "--listen", "stdio://"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to start codex app-server stdio pipes.")

        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._stderr = self._proc.stderr
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        self._initialize()
        self.thread_id = self._start_thread()
        atexit.register(self.close)

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)

    def call(self, system: str, user_message: str, temperature: float = 0.5) -> str:
        prompt = self._compose_prompt(system=system, user_message=user_message, temperature=temperature)
        turn_resp = self._request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": prompt}],
            },
        )

        turn = turn_resp.get("turn", {})
        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            raise RuntimeError("codex app-server returned no turn id.")

        final_text = ""
        fallback_text = ""
        deadline = time.monotonic() + self.config.timeout_seconds

        while True:
            msg = self._read_message(deadline)

            if self._is_server_request(msg):
                self._respond_method_not_supported(msg)
                continue

            method = msg.get("method")
            if method == "item/completed":
                params = msg.get("params", {})
                if params.get("turnId") != turn_id:
                    continue
                item = params.get("item", {})
                if item.get("type") != "agentMessage":
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                fallback_text = text
                if item.get("phase") == "final_answer":
                    final_text = text
                continue

            if method == "turn/completed":
                params = msg.get("params", {})
                turn = params.get("turn", {})
                if turn.get("id") != turn_id:
                    continue
                status = turn.get("status")
                if status != "completed":
                    err = turn.get("error", {})
                    err_msg = err.get("message", "unknown turn failure")
                    raise RuntimeError(f"codex turn failed: {err_msg}")
                output = (final_text or fallback_text).strip()
                if not output:
                    raise RuntimeError("codex app-server returned an empty completion.")
                return output

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "clientInfo": {"name": "loophole", "version": "0.1.0"},
                "capabilities": {
                    # Keeps noisy startup notifications out of the stream.
                    "optOutNotificationMethods": ["mcpServer/startupStatus/updated"],
                },
            },
        )

    def _start_thread(self) -> str:
        result = self._request(
            "thread/start",
            {
                "model": self.config.model,
                "approvalPolicy": self.config.approval_policy,
                "sandbox": self.config.sandbox,
                "ephemeral": self.config.ephemeral,
                "developerInstructions": self.config.developer_instructions,
            },
        )
        thread = result.get("thread", {})
        thread_id = thread.get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise RuntimeError("codex app-server did not return a thread id.")
        return thread_id

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = next(self._id_counter)
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.config.timeout_seconds

        while True:
            msg = self._read_message(deadline)
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(f"codex app-server {method} error: {msg['error']}")
                result = msg.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"codex app-server {method} returned malformed result: {msg}")
                return result

            if self._is_server_request(msg):
                self._respond_method_not_supported(msg)

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(f"codex app-server exited unexpectedly: {self._stderr_summary()}")
        self._stdin.write(json.dumps(payload) + "\n")
        self._stdin.flush()

    def _read_message(self, deadline: float) -> dict[str, Any]:
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(f"codex app-server exited unexpectedly: {self._stderr_summary()}")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for codex app-server response.")

            ready, _, _ = select.select([self._stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("Timed out waiting for codex app-server response.")

            line = self._stdout.readline()
            if line == "":
                raise RuntimeError(f"codex app-server stdout closed: {self._stderr_summary()}")

            raw = line.strip()
            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                # Ignore malformed log lines and keep reading.
                continue
            if isinstance(msg, dict):
                return msg

    def _is_server_request(self, msg: dict[str, Any]) -> bool:
        return (
            "id" in msg
            and "method" in msg
            and "result" not in msg
            and "error" not in msg
        )

    def _respond_method_not_supported(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        if req_id is None:
            return
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": "Client does not support server-initiated requests in this mode.",
                },
            }
        )

    def _drain_stderr(self) -> None:
        for line in self._stderr:
            if line:
                self._stderr_tail.append(line.rstrip())

    def _stderr_summary(self) -> str:
        if not self._stderr_tail:
            return "no stderr output"
        return " | ".join(list(self._stderr_tail)[-3:])

    @staticmethod
    def _compose_prompt(system: str, user_message: str, temperature: float) -> str:
        creativity_note = (
            "Creativity level guidance: "
            f"{temperature:.2f} (0=conservative/precise, 1=creative/exploratory)."
        )
        return (
            "[SYSTEM]\n"
            f"{system}\n\n"
            f"{creativity_note}\n\n"
            "[USER]\n"
            f"{user_message}"
        )


class LLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        provider: str = "anthropic",
        provider_config: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.provider_config = provider_config or {}

        if self.provider == "anthropic":
            self.client = anthropic.Anthropic()
            self.codex_client: CodexAppServerClient | None = None
            return

        if self.provider == "codex_app_server":
            codex_cfg = CodexAppServerConfig(
                command=str(self.provider_config.get("command", "codex")),
                model=str(self.provider_config.get("model", model)),
                sandbox=str(self.provider_config.get("sandbox", "read-only")),
                approval_policy=str(self.provider_config.get("approval_policy", "never")),
                timeout_seconds=float(self.provider_config.get("timeout_seconds", 180.0)),
                ephemeral=bool(self.provider_config.get("ephemeral", True)),
                developer_instructions=str(
                    self.provider_config.get(
                        "developer_instructions",
                        CodexAppServerConfig.developer_instructions,
                    )
                ),
            )
            self.codex_client = CodexAppServerClient(codex_cfg)
            self.client = None
            return

        raise ValueError(
            f"Unsupported provider '{self.provider}'. Expected 'anthropic' or 'codex_app_server'."
        )

    def call(self, system: str, user_message: str, temperature: float = 0.5) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        if self.provider == "codex_app_server" and self.codex_client is not None:
            return self.codex_client.call(system, user_message, temperature=temperature)

        raise RuntimeError(f"Provider '{self.provider}' is not initialized.")

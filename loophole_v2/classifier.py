"""C — the classifier we ship. A small Qwen with a 2-way sequence-classification head.

Input-only: it reads a prompt and predicts {ALLOW, BLOCK}. Each outer round it is retrained
from the base checkpoint on the whole replay buffer, so the buffer *is* its memory and there
is nothing to catastrophically forget. At 0.6B with a few-thousand-example buffer this is
cheap on one H100.

``predict_proba`` drives the adversary's reward; ``embed`` exposes the penultimate
representation for the decision-boundary visualization.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Classifier:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # HF's causal seq-classification head pools the last *non-pad* token assuming
        # right padding, so be explicit about it.
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, dtype=dtype
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device)

    # --- training -----------------------------------------------------------
    def train(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 1,
        lr: float = 1e-5,
        batch_size: int = 16,
        max_len: int = 512,
        optimizer=None,
        step_callback=None,
    ) -> dict:
        """Fine-tune on (texts, labels). Returns final-epoch loss/accuracy.

        Pass ``optimizer`` to warm-start (continuous training across calls); otherwise a fresh
        AdamW is used. ``step_callback`` is called after every optimizer step (e.g. to evaluate
        the held-out set at gradient-step granularity).
        """
        self.model.train()
        opt = optimizer if optimizer is not None else torch.optim.AdamW(self.model.parameters(), lr=lr)
        order = np.arange(len(texts))
        rng = np.random.default_rng(0)

        final_loss, final_acc = 0.0, 0.0
        for _ in range(epochs):
            rng.shuffle(order)
            losses, correct, seen = [], 0, 0
            for start in range(0, len(order), batch_size):
                self.model.train()  # restore train mode (step_callback may have run eval)
                idx = order[start : start + batch_size]
                batch_texts = [texts[i] for i in idx]
                batch_labels = torch.tensor([labels[i] for i in idx], device=self.device)
                enc = self._encode(batch_texts, max_len)
                out = self.model(**enc, labels=batch_labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if step_callback is not None:
                    step_callback()

                losses.append(out.loss.item())
                preds = out.logits.argmax(-1)
                correct += (preds == batch_labels).sum().item()
                seen += len(idx)
            final_loss = float(np.mean(losses))
            final_acc = correct / max(seen, 1)
        return {"train_loss": final_loss, "train_acc": final_acc, "n": len(texts)}

    # --- inference ----------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 32, max_len: int = 512) -> np.ndarray:
        """Return softmax probabilities, shape (N, 2): columns are [p_allow, p_block]."""
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self._encode(texts[start : start + batch_size], max_len)
            logits = self.model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 2))

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 32, max_len: int = 512) -> np.ndarray:
        """Penultimate-layer embedding (last-token hidden state), shape (N, H).

        This is the space C actually decides in, so it's the natural place to view the
        decision boundary.
        """
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self._encode(texts[start : start + batch_size], max_len)
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1]
            last_idx = enc["attention_mask"].sum(dim=1) - 1  # last non-pad (right padding)
            pooled = hidden[torch.arange(hidden.size(0)), last_idx]
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))

    # --- checkpointing ------------------------------------------------------
    def save(self, path: str) -> str:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> "Classifier":
        return cls(model_name=path, device=device, dtype=dtype)

    # --- internals ----------------------------------------------------------
    def _encode(self, texts: list[str], max_len: int) -> dict:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loophole.agents.judge import Judge
from loophole.agents.legislator import Legislator
from loophole.agents.loophole_finder import LoopholeFinder
from loophole.agents.overreach_finder import OverreachFinder
from loophole.llm import LLMClient
from loophole.models import CaseStatus, CaseType, LegalCode, SessionState
from loophole.session import SessionManager


# API Models
class CreateSessionRequest(BaseModel):
    domain: str
    principles: str
    anthropic_api_key: str
    max_rounds: int = 10
    cases_per_agent: int = 3


class UserDecisionRequest(BaseModel):
    case_id: int
    decision: str


class UserVoteRequest(BaseModel):
    case_id: int
    predicted_escalation: bool


class UserSuggestionRequest(BaseModel):
    case_id: int
    suggested_fix: str


class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str = None):
        self.active_connections.remove(websocket)
        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].remove(websocket)

    async def send_to_session(self, session_id: str, message: dict):
        if session_id in self.session_connections:
            disconnected = []
            for connection in self.session_connections[session_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected.append(connection)
            # Clean up disconnected sockets
            for conn in disconnected:
                self.session_connections[session_id].remove(conn)


app = FastAPI(title="Loophole Web API", version="0.1.0")
websocket_manager = WebSocketManager()

# CORS — restricted to local dev origins. Override with LOOPHOLE_CORS_ORIGINS
# env var (comma-separated) for deployments.
_cors_env = os.environ.get("LOOPHOLE_CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Cap on concurrent in-memory sessions (oldest evicted when exceeded).
MAX_ACTIVE_SESSIONS = int(os.environ.get("LOOPHOLE_MAX_SESSIONS", "50"))


class LoopholeWebSession:
    def __init__(self, session_id: str, anthropic_api_key: str, config: dict):
        self.session_id = session_id
        self.config = config

        # Per-session LLM client — API key is passed directly to the Anthropic
        # SDK rather than mutating process env (which would race between sessions).
        self.llm = LLMClient(
            model=config.get("model", "claude-sonnet-4-20250514"),
            max_tokens=config.get("max_tokens", 4096),
            api_key=anthropic_api_key,
        )

        temps = config.get("temperatures", {
            "legislator": 0.4,
            "loophole_finder": 0.9,
            "overreach_finder": 0.9,
            "judge": 0.3
        })
        cases_per = config.get("cases_per_agent", 3)

        self.agents = {
            "legislator": Legislator(self.llm, temperature=temps["legislator"]),
            "loophole": LoopholeFinder(self.llm, temperature=temps["loophole_finder"], cases_per_agent=cases_per),
            "overreach": OverreachFinder(self.llm, temperature=temps["overreach_finder"], cases_per_agent=cases_per),
            "judge": Judge(self.llm, temperature=temps["judge"]),
        }

        self.session_mgr = SessionManager("sessions")
        self.state: Optional[SessionState] = None
        self.user_votes = {}  # case_id -> prediction
        self.user_suggestions = {}  # case_id -> suggestion

    async def create_session(self, domain: str, principles: str):
        """Create new session and generate initial legal code"""
        # Generate initial legal code
        placeholder = SessionState(
            session_id=self.session_id,
            domain=domain,
            moral_principles=principles,
            current_code=LegalCode(version=0, text=""),
        )

        await websocket_manager.send_to_session(self.session_id, {
            "type": "status",
            "message": "Generating initial legal code...",
            "step": "initialization"
        })

        initial_code = self.agents["legislator"].draft_initial(placeholder)
        self.state = self.session_mgr.create_session(self.session_id, domain, principles, initial_code)

        await websocket_manager.send_to_session(self.session_id, {
            "type": "session_created",
            "session": self.state.model_dump(),
            "legal_code": initial_code.model_dump()
        })

        return self.state

    async def run_adversarial_round(self):
        """Run one round of adversarial testing with real-time updates"""
        if not self.state:
            raise ValueError("No active session")

        self.state.current_round += 1
        max_rounds = self.config.get("max_rounds", 10)

        await websocket_manager.send_to_session(self.session_id, {
            "type": "round_started",
            "round": self.state.current_round,
            "max_rounds": max_rounds
        })

        # Phase 1: Find loopholes
        await websocket_manager.send_to_session(self.session_id, {
            "type": "agent_working",
            "agent": "loophole_finder",
            "message": "Searching for loopholes..."
        })

        loopholes = self.agents["loophole"].find(self.state)

        await websocket_manager.send_to_session(self.session_id, {
            "type": "cases_found",
            "agent": "loophole_finder",
            "cases": [case.model_dump() for case in loopholes],
            "count": len(loopholes)
        })

        # Phase 2: Find overreach
        await websocket_manager.send_to_session(self.session_id, {
            "type": "agent_working",
            "agent": "overreach_finder",
            "message": "Searching for overreach..."
        })

        overreaches = self.agents["overreach"].find(self.state)

        await websocket_manager.send_to_session(self.session_id, {
            "type": "cases_found",
            "agent": "overreach_finder",
            "cases": [case.model_dump() for case in overreaches],
            "count": len(overreaches)
        })

        all_cases = loopholes + overreaches

        if not all_cases:
            await websocket_manager.send_to_session(self.session_id, {
                "type": "round_complete",
                "message": "No failures found! Legal code appears robust.",
                "cases_processed": 0
            })
            return

        # Phase 3: Judge each case
        round_stats = {"auto_resolved": 0, "escalated": 0, "total": len(all_cases)}

        for case_obj in all_cases:
            self.state.cases.append(case_obj)

            await websocket_manager.send_to_session(self.session_id, {
                "type": "case_presented",
                "case": case_obj.model_dump()
            })

            # Brief pause for dramatic effect
            await asyncio.sleep(1)

            # Judge evaluates
            await websocket_manager.send_to_session(self.session_id, {
                "type": "judge_evaluating",
                "case_id": case_obj.id
            })

            result = self.agents["judge"].evaluate(self.state, case_obj)

            if result.resolvable:
                # Auto-resolve with validation if needed
                case_obj.resolution = result.resolution_summary or result.reasoning
                case_obj.status = CaseStatus.AUTO_RESOLVED
                case_obj.resolved_by = "judge"

                # Get revised code
                revised = self.agents["legislator"].revise(self.state, case_obj)

                # Validate against existing test suite
                if self.state.resolved_cases:
                    validation = self.agents["judge"].validate(self.state, revised.text)
                    if not validation.passes:
                        case_obj.status = CaseStatus.ESCALATED
                        case_obj.resolution = None
                        case_obj.resolved_by = None

                        await websocket_manager.send_to_session(self.session_id, {
                            "type": "case_escalated",
                            "case": case_obj.model_dump(),
                            "reason": validation.details,
                            "user_vote": self.user_votes.get(case_obj.id),
                            "user_suggestion": self.user_suggestions.get(case_obj.id)
                        })
                        round_stats["escalated"] += 1
                        continue

                # Successfully resolved
                self.state.current_code = revised
                self.state.code_history.append(revised)

                await websocket_manager.send_to_session(self.session_id, {
                    "type": "case_resolved",
                    "case": case_obj.model_dump(),
                    "new_code": revised.model_dump(),
                    "user_vote": self.user_votes.get(case_obj.id)
                })
                round_stats["auto_resolved"] += 1

            else:
                # Escalate to user
                case_obj.status = CaseStatus.ESCALATED
                await websocket_manager.send_to_session(self.session_id, {
                    "type": "case_escalated",
                    "case": case_obj.model_dump(),
                    "reason": result.conflict_explanation or result.reasoning,
                    "user_vote": self.user_votes.get(case_obj.id),
                    "user_suggestion": self.user_suggestions.get(case_obj.id)
                })
                round_stats["escalated"] += 1

            self.session_mgr.save(self.state)

        await websocket_manager.send_to_session(self.session_id, {
            "type": "round_complete",
            "round": self.state.current_round,
            "stats": round_stats,
            "session": self.state.model_dump()
        })

    async def handle_user_decision(self, case_id: int, decision: str):
        """Handle user decision on escalated case"""
        case = next((c for c in self.state.cases if c.id == case_id), None)
        if not case or case.status != CaseStatus.ESCALATED:
            raise ValueError("Invalid case or case not escalated")

        case.status = CaseStatus.USER_RESOLVED
        case.resolution = decision
        case.resolved_by = "user"
        self.state.user_clarifications.append(f"[Case #{case.id}] {decision}")

        # Update legal code
        revised = self.agents["legislator"].revise(self.state, case)
        self.state.current_code = revised
        self.state.code_history.append(revised)

        await websocket_manager.send_to_session(self.session_id, {
            "type": "user_decision_applied",
            "case": case.model_dump(),
            "new_code": revised.model_dump()
        })

        self.session_mgr.save(self.state)


# In-memory session storage (LRU). Sessions are evicted oldest-first once
# MAX_ACTIVE_SESSIONS is reached. Persistence still happens via SessionManager
# to the `sessions/` directory.
active_sessions: "OrderedDict[str, LoopholeWebSession]" = OrderedDict()


def _register_session(session_id: str, session: "LoopholeWebSession") -> None:
    active_sessions[session_id] = session
    active_sessions.move_to_end(session_id)
    while len(active_sessions) > MAX_ACTIVE_SESSIONS:
        active_sessions.popitem(last=False)


def _get_session(session_id: str) -> "LoopholeWebSession":
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    active_sessions.move_to_end(session_id)
    return active_sessions[session_id]


# ============================================================================
# DEMO DATA — sample session for showcasing the UI without API calls
# ============================================================================

def _build_demo_session() -> dict:
    """Pre-baked session demonstrating the full adversarial loop on privacy principles."""
    from datetime import datetime, timedelta
    base = datetime.now() - timedelta(minutes=45)

    principles = (
        "I believe people have a fundamental right to privacy in their personal lives "
        "and communications. Companies should not collect, store, or sell personal data "
        "without explicit, informed consent. Surveillance by the government should require "
        "a warrant based on probable cause. Medical, financial, and communications data "
        "deserve especially strong protection. However, privacy is not absolute — when there "
        "is a specific, credible threat to someone's safety, privacy can be overridden, "
        "but only to the minimum extent necessary."
    )

    code_v1 = (
        "ARTICLE I — CONSENT\n"
        "§1.01 No entity may collect personal data without explicit, informed consent from the subject.\n"
        "§1.02 Consent must be freely given, specific, and revocable at any time.\n\n"
        "ARTICLE II — STATE SURVEILLANCE\n"
        "§2.01 Government surveillance of an individual shall require a warrant issued upon probable cause.\n"
        "§2.02 Mass surveillance without individualized suspicion is prohibited.\n\n"
        "ARTICLE III — SENSITIVE DATA\n"
        "§3.01 Medical, financial, and communications records receive heightened protection.\n"
        "§3.02 Such records may not be disclosed without the subject's written consent.\n\n"
        "ARTICLE IV — EXCEPTIONS\n"
        "§4.01 Privacy may be overridden when necessary to prevent imminent harm to a person."
    )

    code_v2 = (
        "ARTICLE I — CONSENT\n"
        "§1.01 No entity may collect personal data without explicit, informed consent from the subject.\n"
        "§1.02 Consent must be freely given, specific, and revocable at any time.\n"
        "§1.03 [NEW] Consent is not 'informed' where obtained via interface patterns designed to "
        "obscure, rush, or misdirect the subject's choice (commonly: dark patterns). The presence "
        "of pre-ticked boxes, asymmetric visual weight between accept/decline, or nested "
        "disclosures shall void consent.\n\n"
        "ARTICLE II — STATE SURVEILLANCE\n"
        "§2.01 Government surveillance of an individual shall require a warrant issued upon probable cause.\n"
        "§2.02 Mass surveillance without individualized suspicion is prohibited.\n\n"
        "ARTICLE III — SENSITIVE DATA\n"
        "§3.01 Medical, financial, and communications records receive heightened protection.\n"
        "§3.02 Such records may not be disclosed without the subject's written consent.\n\n"
        "ARTICLE IV — EXCEPTIONS\n"
        "§4.01 Privacy may be overridden when necessary to prevent imminent harm to a person."
    )

    code_v3 = (
        "ARTICLE I — CONSENT\n"
        "§1.01 No entity may collect personal data without explicit, informed consent from the subject.\n"
        "§1.02 Consent must be freely given, specific, and revocable at any time.\n"
        "§1.03 Consent is not 'informed' where obtained via interface patterns designed to "
        "obscure, rush, or misdirect the subject's choice. The presence of pre-ticked boxes, "
        "asymmetric visual weight between accept/decline, or nested disclosures shall void consent.\n\n"
        "ARTICLE II — STATE SURVEILLANCE\n"
        "§2.01 Government surveillance of an individual shall require a warrant issued upon probable cause.\n"
        "§2.02 Mass surveillance without individualized suspicion is prohibited.\n\n"
        "ARTICLE III — SENSITIVE DATA\n"
        "§3.01 Medical, financial, and communications records receive heightened protection.\n"
        "§3.02 Such records may not be disclosed without the subject's written consent.\n\n"
        "ARTICLE IV — EXCEPTIONS\n"
        "§4.01 Privacy may be overridden when necessary to prevent imminent harm to a person.\n"
        "§4.02 [NEW] A licensed medical provider rendering emergency care may access existing "
        "medical records without prior consent where (a) the subject is incapacitated, and "
        "(b) access is necessary for the immediate treatment decision at hand. Such access "
        "must be logged and reviewable by the subject upon recovery."
    )

    code_v4 = (
        "ARTICLE I — CONSENT\n"
        "§1.01 No entity may collect personal data without explicit, informed consent from the subject.\n"
        "§1.02 Consent must be freely given, specific, and revocable at any time.\n"
        "§1.03 Consent is not 'informed' where obtained via interface patterns designed to "
        "obscure, rush, or misdirect the subject's choice. The presence of pre-ticked boxes, "
        "asymmetric visual weight between accept/decline, or nested disclosures shall void consent.\n"
        "§1.04 [NEW] Aggregation of individually anonymized datasets constitutes a new act of "
        "collection where the combined dataset is reasonably re-identifiable. The entity "
        "performing the aggregation bears the burden of demonstrating non-re-identifiability.\n\n"
        "ARTICLE II — STATE SURVEILLANCE\n"
        "§2.01 Government surveillance of an individual shall require a warrant issued upon probable cause.\n"
        "§2.02 Mass surveillance without individualized suspicion is prohibited.\n\n"
        "ARTICLE III — SENSITIVE DATA\n"
        "§3.01 Medical, financial, and communications records receive heightened protection.\n"
        "§3.02 Such records may not be disclosed without the subject's written consent.\n\n"
        "ARTICLE IV — EXCEPTIONS\n"
        "§4.01 Privacy may be overridden when necessary to prevent imminent harm to a person.\n"
        "§4.02 A licensed medical provider rendering emergency care may access existing "
        "medical records without prior consent where (a) the subject is incapacitated, and "
        "(b) access is necessary for the immediate treatment decision at hand."
    )

    cases = [
        {
            "id": 1, "round": 1, "case_type": "loophole",
            "scenario": (
                "A social media company displays a consent dialog with 'Accept All' rendered "
                "as a large, colorful, center-positioned button, while 'Manage Preferences' "
                "appears as low-contrast text in a corner. The user clicks Accept within two "
                "seconds, without reading the 47-page terms document linked at the bottom."
            ),
            "explanation": (
                "§1.01 requires 'explicit, informed consent' but the code does not define what "
                "makes consent 'informed.' The company can argue the user clicked the button, "
                "satisfying the explicit requirement — while exploiting interface design to "
                "ensure the consent is never meaningfully informed. Technically legal, "
                "morally fraudulent."
            ),
            "status": "auto_resolved", "resolved_by": "judge",
            "resolution": (
                "Narrow §1.01 by defining 'informed' to exclude consent obtained through "
                "manipulative interface design. No prior precedent is contradicted."
            ),
            "created_at": (base + timedelta(minutes=2)).isoformat()
        },
        {
            "id": 2, "round": 1, "case_type": "overreach",
            "scenario": (
                "A 34-year-old arrives unconscious at a hospital emergency room after a "
                "motorcycle accident. The attending physician needs to check the patient's "
                "medical records for drug allergies before administering anesthesia. The "
                "patient cannot provide written consent."
            ),
            "explanation": (
                "§3.02 prohibits disclosure of medical records without the subject's 'written "
                "consent.' The emergency exception in §4.01 covers preventing 'imminent harm' "
                "but is framed around overriding privacy against a threatening third party, "
                "not about the subject's own medical care. Under a strict reading, the physician "
                "must either delay treatment or violate the code."
            ),
            "status": "auto_resolved", "resolved_by": "judge",
            "resolution": (
                "Add §4.02 creating a narrow emergency medical exception with procedural "
                "safeguards (logging, post-hoc review). Consistent with §4.01's existing "
                "proportionality principle."
            ),
            "created_at": (base + timedelta(minutes=8)).isoformat()
        },
        {
            "id": 3, "round": 2, "case_type": "loophole",
            "scenario": (
                "A data broker purchases ten separately anonymized datasets — browsing history, "
                "location pings, purchase records, etc. — each legally collected under §1.01. "
                "By cross-referencing them, the broker can re-identify 89% of individuals and "
                "sell detailed personal profiles to insurers and employers."
            ),
            "explanation": (
                "Each individual dataset is legal: it was anonymized and lawfully collected. "
                "The code never anticipated that the *combination* of lawful datasets could "
                "constitute a new, more invasive act. The broker exploits the gap between "
                "individual compliance and aggregate harm."
            ),
            "status": "auto_resolved", "resolved_by": "judge",
            "resolution": (
                "Add §1.04 treating aggregation of datasets as a new act of collection where "
                "re-identification is reasonably possible, placing burden of proof on the aggregator."
            ),
            "created_at": (base + timedelta(minutes=19)).isoformat()
        },
        {
            "id": 4, "round": 2, "case_type": "overreach",
            "scenario": (
                "An investigative journalist obtains leaked internal emails from a pharmaceutical "
                "executive revealing that the company knowingly suppressed data about a drug's "
                "side effects. Publishing requires disclosing communications data without the "
                "executive's consent."
            ),
            "explanation": (
                "§3.01 grants heightened protection to 'communications data.' §3.02 prohibits "
                "disclosure without written consent. No exception in the current code covers "
                "journalism, whistleblowing, or public-interest disclosure of corporate "
                "wrongdoing. The code would protect the wrongdoer and silence the press."
            ),
            "status": "escalated", "resolved_by": None, "resolution": None,
            "created_at": (base + timedelta(minutes=27)).isoformat()
        },
        {
            "id": 5, "round": 2, "case_type": "loophole",
            "scenario": (
                "A municipal government deploys a network of license-plate readers on public "
                "roads. Each individual reading is not 'surveillance of an individual' — it's "
                "reading a publicly-displayed plate. But the database allows police to "
                "reconstruct anyone's movements over months without a warrant."
            ),
            "explanation": (
                "§2.01 requires a warrant for 'surveillance of an individual.' The city argues "
                "each reading is a discrete observation of a public object, not surveillance of "
                "a person. §2.02 prohibits 'mass surveillance' but the city calls this "
                "'traffic monitoring.' The aggregation loophole from Case #3 is recurring here, "
                "but in a governmental context."
            ),
            "status": "pending", "resolved_by": None, "resolution": None,
            "created_at": (base + timedelta(minutes=34)).isoformat()
        },
    ]

    code_history = [
        {"version": 1, "text": code_v1, "changelog": "Initial draft from stated principles",
         "created_at": base.isoformat()},
        {"version": 2, "text": code_v2,
         "changelog": "Case #1 — Added §1.03 defining 'informed' consent to exclude dark patterns",
         "created_at": (base + timedelta(minutes=4)).isoformat()},
        {"version": 3, "text": code_v3,
         "changelog": "Case #2 — Added §4.02 emergency medical exception with logging safeguards",
         "created_at": (base + timedelta(minutes=10)).isoformat()},
        {"version": 4, "text": code_v4,
         "changelog": "Case #3 — Added §1.04 treating dataset aggregation as re-collection",
         "created_at": (base + timedelta(minutes=21)).isoformat()},
    ]

    return {
        "session_id": "demo_privacy_exhibit",
        "domain": "privacy",
        "moral_principles": principles,
        "user_clarifications": [],
        "current_code": code_history[-1],
        "code_history": code_history,
        "cases": cases,
        "current_round": 2,
        "created_at": base.isoformat(),
    }


@app.get("/api/demo")
async def get_demo_session():
    """Return a pre-baked sample session for UI demonstration — no API key required."""
    return _build_demo_session()


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new Loophole session"""
    session_id = f"{request.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperatures": {
            "legislator": 0.4,
            "loophole_finder": 0.9,
            "overreach_finder": 0.9,
            "judge": 0.3,
        },
        "max_rounds": request.max_rounds,
        "cases_per_agent": request.cases_per_agent,
    }

    web_session = LoopholeWebSession(session_id, request.anthropic_api_key, config)
    _register_session(session_id, web_session)

    try:
        state = await web_session.create_session(request.domain, request.principles)
        return {"session_id": session_id, "state": state.model_dump()}
    except Exception as e:
        active_sessions.pop(session_id, None)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/round")
async def start_round(session_id: str):
    """Start a new adversarial round"""
    web_session = _get_session(session_id)
    await web_session.run_adversarial_round()
    return {"message": "Round started"}


@app.post("/api/sessions/{session_id}/decisions")
async def submit_decision(session_id: str, request: UserDecisionRequest):
    """Submit user decision for escalated case"""
    web_session = _get_session(session_id)
    await web_session.handle_user_decision(request.case_id, request.decision)
    return {"message": "Decision applied"}


@app.post("/api/sessions/{session_id}/votes")
async def submit_vote(session_id: str, request: UserVoteRequest):
    """Submit user vote/prediction for case outcome"""
    web_session = _get_session(session_id)
    web_session.user_votes[request.case_id] = request.predicted_escalation
    return {"message": "Vote recorded"}


@app.post("/api/sessions/{session_id}/suggestions")
async def submit_suggestion(session_id: str, request: UserSuggestionRequest):
    """Submit user suggestion for case resolution"""
    web_session = _get_session(session_id)
    web_session.user_suggestions[request.case_id] = request.suggested_fix
    return {"message": "Suggestion recorded"}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get current session state"""
    return _get_session(session_id).state.model_dump()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Release an in-memory session (does not delete persisted session data)."""
    active_sessions.pop(session_id, None)
    return {"message": "Session released"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
"""
FastAPI backend that wraps the boto3 `bedrock-agentcore` call.

Why this exists: boto3 (and AWS credentials) cannot run in the browser.
React talks to this small API over HTTP; this API is the only place that
holds AWS credentials and calls invoke_agent_runtime.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
"""

import json
import os
import uuid

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN", "")
ENDPOINT_QUALIFIER = os.getenv("ENDPOINT_QUALIFIER") or None
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

if not AGENT_RUNTIME_ARN:
    raise RuntimeError(
        "AGENT_RUNTIME_ARN is not set. Copy backend/.env.example to backend/.env "
        "and fill in your values."
    )

# boto3 will pick up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
# from the environment automatically if you set them in .env (loaded above),
# or fall back to your normal AWS credential chain (~/.aws/credentials, an
# instance role, etc.) if you leave them unset. Either way works.
client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

app = FastAPI(title="AgentCore Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # if omitted, a new one is generated


class ChatResponse(BaseModel):
    session_id: str
    response: str
    raw: dict | None = None


def _ensure_session_id(session_id: str | None) -> str:
    # runtimeSessionId must be 33+ characters
    if session_id and len(session_id) >= 33:
        return session_id
    return str(uuid.uuid4()) + str(uuid.uuid4())[:2]  # 38 chars, always long enough


def _build_invoke_kwargs(session_id: str, message: str) -> dict:
    invoke_kwargs = dict(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": message}),
    )
    if ENDPOINT_QUALIFIER:
        invoke_kwargs["qualifier"] = ENDPOINT_QUALIFIER
    return invoke_kwargs


def _extract_text(response_data: dict) -> str:
    """Best-effort extraction of the assistant's reply text.
    Adjust this once you know the exact shape your agent returns."""
    if isinstance(response_data, str):
        return response_data
    for key in ("response", "completion", "output", "result", "text", "message"):
        if key in response_data:
            value = response_data[key]
            return value if isinstance(value, str) else json.dumps(value)
    return json.dumps(response_data)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = _ensure_session_id(req.session_id)
    invoke_kwargs = _build_invoke_kwargs(session_id, req.message)

    try:
        response = client.invoke_agent_runtime(**invoke_kwargs)
        response_body = response["response"].read()
        response_data = json.loads(response_body)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail=f"AgentCore invoke failed: {exc}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="AgentCore returned non-JSON response")

    return ChatResponse(
        session_id=session_id,
        response=_extract_text(response_data),
        raw=response_data if isinstance(response_data, dict) else None,
    )


def _sse(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    # each line of the payload needs its own "data: " prefix per the SSE spec
    payload = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"{prefix}{payload}\n\n"


def _stream_agent_reply(session_id: str, message: str):
    """Yields SSE frames as the agent's reply becomes available.

    Handles both cases AWS documents for invoke_agent_runtime:
    - contentType 'text/event-stream': the agent is itself streaming tokens,
      forward each one as it arrives.
    - contentType 'application/json' (or anything else): the agent returned
      a single response, so it's emitted as one chunk.
    """
    invoke_kwargs = _build_invoke_kwargs(session_id, message)
    response = client.invoke_agent_runtime(**invoke_kwargs)
    content_type = response.get("contentType", "")

    if "text/event-stream" in content_type:
        for line in response["response"].iter_lines(chunk_size=1):
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                decoded = decoded[len("data: "):]
            # NOTE: this assumes each event's data is plain text (a token).
            # If your agent emits JSON per event instead, parse it here and
            # pull out the text field before yielding.
            yield _sse(decoded)
    elif content_type == "application/json":
        body = b"".join(chunk for chunk in response.get("response", []))
        try:
            text = _extract_text(json.loads(body))
        except json.JSONDecodeError:
            text = body.decode("utf-8", errors="replace")
        yield _sse(text)
    else:
        body = response["response"].read()
        try:
            text = _extract_text(json.loads(body))
        except json.JSONDecodeError:
            text = body.decode("utf-8", errors="replace")
        yield _sse(text)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    session_id = _ensure_session_id(req.session_id)

    def event_generator():
        yield _sse(session_id, event="session")
        try:
            for frame in _stream_agent_reply(session_id, req.message):
                yield frame
        except (BotoCoreError, ClientError) as exc:
            yield _sse(str(exc), event="error")
        finally:
            yield _sse("end", event="done")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

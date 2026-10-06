import json
import os
import uuid

import boto3
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# AgentCore runtime config (set these in your environment or a .env file)
# ---------------------------------------------------------------------------
REGION = os.getenv("AWS_REGION")
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
QUALIFIER = os.getenv("AGENT_RUNTIME_QUALIFIER")  # optional; unset/empty to omit

_client = boto3.client("bedrock-agentcore", region_name=REGION)


def new_session_id() -> str:
    # runtimeSessionId must be 33+ chars
    return str(uuid.uuid4()) + str(uuid.uuid4())


def invoke_agent(prompt: str, session_id: str) -> str:
    payload = json.dumps({"prompt": prompt})

    kwargs = dict(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=session_id,
        payload=payload,
    )
    if QUALIFIER:
        kwargs["qualifier"] = QUALIFIER

    response = _client.invoke_agent_runtime(**kwargs)
    response_body = response["response"].read()
    response_data = json.loads(response_body)

    # main.py's entrypoint returns {"result": output}
    return response_data.get("result", response_data)
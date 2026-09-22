import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")


def get_weather(city: str) -> str:
    """Fetch current weather from the configured weather API."""
    api_key = os.getenv("WEATHERAPI_KEY")

    if not WEATHER_API_URL:
        return json.dumps(
            {"success": False, "message": "WEATHER_API_URL is not configured"}
        )
    if not api_key:
        return json.dumps(
            {"success": False, "message": "WEATHERAPI_KEY is not configured"}
        )

    url = f"{WEATHER_API_URL}?{urlencode({'key': api_key, 'q': city.strip()})}"

    try:
        with urlopen(url, timeout=10) as response:
            weather_data = json.load(response)
    except HTTPError as error:
        return json.dumps(
            {"success": False, "status_code": error.code, "message": error.reason}
        )
    except (URLError, TimeoutError) as error:
        return json.dumps({"success": False, "message": str(error)})

    return json.dumps(
        {
            "success": True,
            "location": weather_data.get("location"),
            "current": weather_data.get("current"),
        }
    )


tool_functions = {"get_weather": get_weather}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def ask_weather(user_query):
    messages = [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:
        return message.content

    for tool_call in message.tool_calls:
        function = tool_functions.get(tool_call.function.name)
        if function is None:
            result = json.dumps({"success": False, "message": "Unknown tool"})
        else:
            arguments = json.loads(tool_call.function.arguments)
            result = function(**arguments)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask_weather("What's the weather in Mumbai?"))

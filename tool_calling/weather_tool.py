import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

WEATHER_API_URL = os.getenv("WEATHER_API_URL")


@tool
def get_current_weather(city: str) -> str:
    """
    Get the current weather for a city.
    The city argument should be a city name, such as "London" or "New York".
    """

    city = city.strip()
    api_key = os.getenv("WEATHERAPI_KEY")

    if not city:
        return json.dumps(
            {
                "success": False,
                "message": "city must not be empty",
            }
        )

    if not api_key:
        return json.dumps(
            {
                "success": False,
                "message": "WEATHERAPI_KEY is not configured",
            }
        )

    url = f"{WEATHER_API_URL}?{urlencode({'key': api_key, 'q': city})}"

    try:
        with urlopen(url, timeout=10) as response:
            weather_data = json.load(response)
    except HTTPError as error:
        try:
            error_data = json.load(error)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_data = {"message": error.reason}

        return json.dumps(
            {
                "success": False,
                "status_code": error.code,
                "error": error_data.get("error", error_data),
            }
        )
    except (URLError, TimeoutError) as error:
        return json.dumps(
            {
                "success": False,
                "message": f"Weather service request failed: {error.reason if isinstance(error, URLError) else error}",
            }
        )

    return json.dumps(
        {
            "success": True,
            "location": weather_data.get("location"),
            "current": weather_data.get("current"),
        }
    )

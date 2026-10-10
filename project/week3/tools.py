import json
import os
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen
from langchain_community.tools import DuckDuckGoSearchResults

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

WEATHER_API_URL = os.getenv("WEATHER_API_URL")

web_search_tool = DuckDuckGoSearchResults(max_results=5, output_format="json")
web_search_tool.description = (
    "Search the live web for CURRENT information. Returns results with title, snippet and link; cite "
    "the links. Input: a specific search query."
)


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a city."""

    city = city.strip()
    api_key = os.getenv("WEATHERAPI_KEY")

    if not city:
        return json.dumps({"success": False, "message": "city must not be empty"})

    if not api_key:
        return json.dumps(
            {"success": False, "message": "WEATHERAPI_KEY is not configured"}
        )

    try:
        response = requests.get(
            WEATHER_API_URL, params={"key": api_key, "q": city}, timeout=10
        )
        data = response.json()

    except requests.RequestException as error:
        return json.dumps(
            {
                "success": False,
                "message": f"Weather service request failed ({type(error).__name__})",
            }
        )

    except ValueError:
        # response.json() raises ValueError when the body isn't valid JSON.
        return json.dumps(
            {
                "success": False,
                "status_code": response.status_code,
                "message": "Weather service returned invalid JSON",
            }
        )
    if not response.ok:
        return json.dumps(
            {
                "success": False,
                "status_code": response.status_code,
                "error": data.get("error", data),
            }
        )
    return json.dumps(
        {
            "success": True,
            "location": data.get("location"),
            "current": data.get("current"),
        }
    )


@tool
def wikipedia_lookup(topic: str) -> str:
    """Fetch a short factual summary for a topic from Wikipedia."""
    try:
        page = quote(topic.strip().replace(" ", "_"), safe="")
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + page,
            headers={"User-Agent": "query-agent/1.0"},
            timeout=10,
        )
        if r.status_code != 200:
            return f"No Wikipedia page found for '{topic}'."
        return r.json().get("extract", "No summary available.")
    except Exception as e:
        return f"Lookup failed: {e}"


@tool
def search_employee_handbook(query: str) -> str:
    """Find relevant information in the indexed employee handbook."""
    query = query.strip()
    if not query:
        return "Please provide a question or search phrase."

    from vector_store import PDF_PATH, embed, get_vector_store

    store = get_vector_store()
    try:
        has_data = bool(store.get_pks(expr="pk >= 0"))
    except Exception:
        has_data = False

    if not has_data:
        embed(PDF_PATH)
        store = get_vector_store()

    matches = store.similarity_search(query, k=4)
    if not matches:
        return "No relevant handbook information was found."

    results = []
    for index, document in enumerate(matches, start=1):
        page = document.metadata.get("page")
        page_reference = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
        results.append(f"[{index}] {page_reference}\n{document.page_content.strip()}")

    return "\n\n".join(results)


tools = [
    get_current_weather,
    wikipedia_lookup,
    web_search_tool,
    search_employee_handbook,
]

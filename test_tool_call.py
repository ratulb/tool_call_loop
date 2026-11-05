"""A minimal local tool-calling loop: one model, one tool, one real HTTP call.

The model never runs the tool. It emits a request to call it; this script
executes the function and sends the result back so the model can answer.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

from openai import OpenAI

MODEL = "refinedneuro/refinedtoolcallv5-3b"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def get_weather(city: str) -> str:
    """Current temperature for a city, from the free Open-Meteo API."""
    try:
        query = urllib.parse.urlencode({"name": city, "count": 1})
        geo = fetch_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}")
        if not geo.get("results"):
            return f"Could not find {city}."
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        forecast = fetch_json(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        )
    except (urllib.error.URLError, TimeoutError) as error:
        return f"The weather service could not be reached: {error}"
    temp = forecast["current"]["temperature_2m"]
    code = forecast["current"]["weather_code"]
    return f"The current temperature in {city} is {temp} deg C (weather code: {code})."


# The only functions the model is allowed to trigger.
TOOL_FUNCTIONS = {"get_weather": get_weather}

# What the model sees: a JSON-Schema description of each tool.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }
]

ANSWER_PROMPT = (
    "You are a helpful assistant. The tool result above is real, live data "
    "fetched from a weather API. Use it to answer the user's question directly "
    "and concisely. Do not say you cannot access external APIs. Do not write code."
)


def run_tool_call(call) -> str:
    """Execute one tool call requested by the model and return the result text."""
    function = TOOL_FUNCTIONS.get(call.function.name)
    if function is None:
        return f"Unknown tool: {call.function.name}"
    try:
        args = json.loads(call.function.arguments)  # arrives as a JSON string
        return function(**args)
    except (json.JSONDecodeError, TypeError) as error:
        return f"Bad arguments for {call.function.name}: {error}"


def main() -> None:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by the SDK, ignored by Ollama
    )

    messages = [{"role": "user", "content": "What is the weather in Bengaluru right now?"}]

    # Turn 1: the model decides whether to request a tool call.
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0.6,
        top_p=0.95,
    )
    message = response.choices[0].message

    if not message.tool_calls:
        print("No tool call was made. The model answered directly:")
        print(message.content)
        return

    # The model must see its own request next to the result it gets back.
    messages.append(message)
    for call in message.tool_calls:
        print(f"Model requested: {call.function.name}({call.function.arguments})")
        result = run_tool_call(call)
        print(f"Tool result: {result}")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Turn 2: no tools offered; the model now answers from the result.
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": ANSWER_PROMPT}, *messages],
        temperature=0.3,
    )
    print("\nFinal answer from model:")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    main()
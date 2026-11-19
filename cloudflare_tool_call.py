"""A minimal local tool-calling loop on Cloudflare Workers AI: one model, one tool,
one real HTTP call.

The model never runs the tool. It emits a request to call it; this script
executes the function and sends the result back so the model can answer.

Uses Workers AI's OpenAI-compatible endpoint, so the OpenAI SDK works unchanged
apart from base_url, api_key and the model name.

Required environment variables:
    CLOUDFLARE_ACCOUNT_ID
    CLOUDFLARE_AUTH_TOKEN   (API token with Workers AI access)
"""
import os
import json
import urllib.error
import urllib.parse
import urllib.request

from openai import OpenAI

# The model must support function calling (look for the "function calling"
# property in the Workers AI model catalog), otherwise the API will reject the
# `tools` field or the model will never emit a tool call.
#MODEL = "@cf/qwen/qwen3.8-27b"
#MODEL = "@cf/google/gemma-4-26b-a4b-it"
#MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
#MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"
#MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
#MODEL = "@cf/openai/gpt-oss-120b"
#MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
#MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"
#MODEL = "@cf/zai-org/glm-4.7-flash"
#MODEL = "@cf/ibm-granite/granite-4.0-h-micro"
#MODEL = "@cf/openai/gpt-oss-20b"
#MODEL = "@cf/nvidia/nemotron-3-120b-a12b"
#MODEL = "@cf/daily-co/phonellm-mini-alpha-2"
MODEL = "@cf/qwen/qwq-32b"

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
        args = call.function.arguments
        # OpenAI format sends a JSON string; tolerate an already-parsed dict too.
        if isinstance(args, str):
            args = json.loads(args)
        return function(**args)
    except (json.JSONDecodeError, TypeError) as error:
        return f"Bad arguments for {call.function.name}: {error}"


def assistant_message_to_dict(message) -> dict:
    """Plain-dict copy of the assistant's tool request.

    Sending back a minimal dict (instead of the SDK's response object) avoids
    passing provider-specific extra fields back to the server.
    """
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": (
                        call.function.arguments
                        if isinstance(call.function.arguments, str)
                        else json.dumps(call.function.arguments)
                    ),
                },
            }
            for call in message.tool_calls
        ],
    }


def main() -> None:
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    client = OpenAI(
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        api_key=os.environ["CLOUDFLARE_AUTH_TOKEN"],
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
    messages.append(assistant_message_to_dict(message))
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

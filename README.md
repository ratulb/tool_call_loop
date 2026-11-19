# tool-call-loop

A minimal tool-calling loop for a local LLM, in about 100 lines of Python. One model, one tool, one real HTTP call, one answer, end to end on your own machine.

The model runs locally on CPU, with no account and no API key. (You do need a network connection: once to download the model, and each run to reach the free weather API.)

The repo exists to make one primitive concrete: a local model emits a structured request to call a tool, your code executes the function, the result goes back to the model, and the model answers from it. It also explains the tooling around that loop (Ollama, GGUF, llama.cpp, quantization, vLLM), because the point is not "it worked" but "here is the whole stack, and what each piece does."

## What it does

```text
user question
    │
    ▼
┌─────────────────┐     tool call (JSON)     ┌───────────────────┐
│  local LLM      │ ───────────────────────► │  Python code      │
│ (Ollama, 3B)    │                          │  get_weather()    │
└─────────────────┘                          └─────────┬─────────┘
    ▲                                                  │
    │        tool result (role: "tool")                │ real HTTP
    └──────────────────────────────────────────────────┘
    │
    ▼
final answer (prose, grounded in the tool result)
```

**The model does not call the tool. The model emits a request to call the tool, and your code is the executor.** That boundary is the architecture of every agent, and this repo is the smallest demonstration of it.

## What you need

- Python 3.9 or newer
- About 4 GB of free RAM for the 3B model
- Linux or macOS (the commands below use a POSIX shell)
- No GPU, no account, no API key

## Quick start

### 1. Install Ollama

Ollama is a local model runner. It downloads models, serves them over an HTTP API, and stays out of your way.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama -v
```

It serves on `http://localhost:11434`, including an OpenAI-compatible endpoint under `/v1`. That is why the code below uses the standard `openai` client with a different `base_url`.

### 2. Pull the model

This repo uses [`refinedneuro/refinedtoolcallv5-3b`](https://ollama.com/refinedneuro/refinedtoolcallv5-3b): a 3B-parameter model tuned for multi-turn tool calling. It is about 2.5 GB, runs on CPU, and is licensed Apache-2.0.

```bash
ollama pull refinedneuro/refinedtoolcallv5-3b
```

A bare name such as `llama3.2` comes from the public [Ollama library](https://ollama.com/library). A `user/model` path, as above, comes from that user's namespace in the same registry.

### 3. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai
```

Only `openai` is required. The weather call uses the standard library (`urllib`, `json`), so there is no `requests` dependency.

### 4. Run it

```bash
python test_tool_call.py
```

Example output (yours will differ, because sampling is random and the weather changes):

```text
Model requested: get_weather({"city": "Bengaluru"})
Tool result: The current temperature in Bengaluru is 26.6 deg C (weather code: 53).

Final answer from model:
The current temperature in Bengaluru is 26.6°C with drizzle.
```

That is the loop.

## The script

`test_tool_call.py`:

```python
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
```

## Reading the script

| Piece | What it does |
| --- | --- |
| `OpenAI(base_url="http://localhost:11434/v1", ...)` | Points the OpenAI SDK at the local Ollama server instead of `api.openai.com`. Ollama implements the same `/v1/chat/completions` shape, so the SDK works unchanged. |
| `api_key="ollama"` | The SDK requires a key; Ollama ignores it. Any non-empty string works. |
| `TOOLS` | The tool schema: a JSON-Schema description of the function (name, purpose, parameters). The model reads it to decide what to call and how. |
| `messages = [...]` | The conversation lives in a variable because it is appended to later: first the model's request, then the tool's result. |
| `temperature=0.6, top_p=0.95` | Standard sampling settings for turn 1. Treat them as a starting point and check the model page for its recommendation. |
| `message.tool_calls` | Populated if the model asked for a tool; otherwise the reply is plain prose in `message.content`. |
| `messages.append(message)` | Adds the model's own request to the conversation. The model must see it to match the result to it. |
| `call.function.arguments` | The arguments arrive as a JSON **string**, not a dict. Parse it before use, and expect it to be malformed sometimes. |
| `TOOL_FUNCTIONS[...]` | Your code decides what runs. The model supplies a name and arguments; it cannot execute anything. |
| `{"role": "tool", "tool_call_id": ..., "content": ...}` | Sends the result back. `tool_call_id` is mandatory: it links the result to the specific call that requested it. |
| Second `create(...)` | Turn 2. No `tools=`, because the model is now answering, not calling. |
| System message on turn 2 | Tells a small model that the tool result is real and to answer from it. Without it, a 3B model often says "I cannot access live data" although the result is in its context. |
| `temperature=0.3` on turn 2 | Lowers variance so the model is less likely to wander off the result. |

## What this exercise teaches

### 1. The loop is trivial; the reliability is not

The plumbing (emit call, execute, return result, answer) is a few dozen lines, which is why agents are easy to demo. What is hard is making the answer trustworthy.

### 2. Watch the model hallucinate around the tool result

The weather API returns two fields: `temperature_2m` and `weather_code`. In one test run, the model used the temperature correctly, then invented humidity, wind speed, sky conditions, sunset time and a forecast, none of which were in the result. It also read weather code 53 as "sunny with scattered clouds". In Open-Meteo's WMO code table, 53 is moderate drizzle.

A working tool call does not mean a working answer. Output validation, constrained prompting and evaluation are what make a tool-using system dependable. That is the layer above this repo.

### 3. The same API can mean different things on different servers

The OpenAI chat API has a `tool_choice` parameter that can force a particular function. Ollama's OpenAI-compatible endpoint [does not support it](https://docs.ollama.com/api/openai-compatibility), so this script cannot force a call: the model decides, and turn 1 may come back as prose, which the script handles. Sampling options that Ollama does support (such as `temperature`, `top_p` and `seed`) go through the standard fields; Ollama-specific ones such as `repeat_penalty` belong to its native API or a Modelfile. "OpenAI-compatible" means a subset of the API, so check what your server honours before relying on a parameter.

### 4. Model output is untrusted input

The model chooses the function name and arguments. The script runs only functions in an allow-list (`TOOL_FUNCTIONS`), parses the arguments defensively, and URL-encodes the city before it goes into a request. With a more powerful tool, such as a shell or a database, that discipline stops being optional.

## The tooling landscape

Working in this layer means knowing four pieces.

**Ollama.** A local model runner and server. It pulls models from its registry ([ollama.com/library](https://ollama.com/library)), stores them locally, serves them on `http://localhost:11434` (including the OpenAI-compatible `/v1` endpoint), and handles model loading, context and sampling. It is the easiest way to run a model locally. It is built for one user on one machine, not for high-concurrency serving.

**GGUF.** The single-file format used by llama.cpp, and so by Ollama for most models. One file bundles the weights, the tokenizer and metadata, and the quantization scheme. When you `ollama pull` a model, you are pulling GGUF data, not a PyTorch checkpoint.

**llama.cpp.** The C++ inference engine that Ollama was built on. It runs GGUF models on CPU, and on GPU with the right backend, and it provides the conversion and quantization tools: `convert_hf_to_gguf.py` turns a Hugging Face checkpoint into a 16-bit GGUF, and `llama-quantize` compresses it to a chosen level (`Q4_K_M`, `Q6_K`, `Q8_0`, and so on). Newer Ollama versions also ship their own engine for some models.

**Who quantizes.** Usually not the lab that trained the model. Labs typically release 16-bit `safetensors` weights. The conversion and quantization are done by the Ollama team for models in its library, by community quantizers on Hugging Face, or by you. Ollama can also quantize on import (`ollama create --quantize`), using the same families of schemes. If a model is not in the registry, convert it with llama.cpp's tools and import the file with a `Modelfile`.

**vLLM.** A production serving engine, aimed mainly at GPUs. Where Ollama is optimised for one model and one user, vLLM is optimised for many concurrent requests: continuous batching and PagedAttention for KV-cache efficiency. If this loop became a service, vLLM or an equivalent is what you would serve the model with. It is not needed here.

In one line: llama.cpp is the engine, GGUF is the file format it runs, Ollama is the friendly wrapper, and vLLM is what you reach for when this becomes a service.

## Files

```text
tool-call-loop/
├── test_tool_call.py   # the loop
└── README.md           # this file
```

## What's next (deliberately parked)

The loop works. The next layer is retrieval, multiple tools and evaluation, which is where the hallucination described above has to be caught. This repo stops at the primitive on purpose.

## Notes and limits

- The model, its size and its licence are those reported on its Ollama page; check it before you build on it.
- Open-Meteo is free for non-commercial use and needs no key; see [its terms](https://open-meteo.com/en/terms) before using it any other way.
- The script handles several tool calls in one reply, but the demo question needs only one.

## References

- Ollama: <https://ollama.com>
- Ollama library: <https://ollama.com/library>
- Ollama OpenAI compatibility: <https://docs.ollama.com/api/openai-compatibility>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- GGUF format: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- vLLM: <https://github.com/vllm-project/vllm>
- Open-Meteo: <https://open-meteo.com>

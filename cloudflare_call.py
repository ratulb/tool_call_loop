import os
import requests


ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
API_TOKEN = os.environ["CLOUDFLARE_AUTH_TOKEN"]


response = requests.post(
    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/"
    "@cf/qwen/qwen3.8-27b",
    headers={
        "Authorization": f"Bearer {API_TOKEN}"
    },
    json={
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function that reverses "
                    "a string without using slicing."
                ),
            }
        ],
        "max_tokens": 1024,
    },
)

print(response.json()["result"]["choices"][0]["message"]["content"])

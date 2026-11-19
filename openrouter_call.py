import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning:free",
    messages=[
        {
            "role": "user",
            #"content": "How many r's are in the word 'strawberry'?"
            "content": "What is the meaning of life?"
        }
    ],
    extra_body={
        "reasoning": {
            "enabled": True
        }
    },
)

message = response.choices[0].message

print(message.reasoning)
print(message.content)

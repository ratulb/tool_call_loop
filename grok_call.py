from groq import Groq

# Create a client instance with your API key
# Make sure you set your GROQ_API_KEY environment variable before running
# Example: export GROQ_API_KEY="your_api_key_here"
#client = Groq(api_key="your_api_key_here")  # Or omit api_key to use env var
client = Groq()

try:
    # Example: Call the Groq API to generate text
    response = client.chat.completions.create(
        #model="llama3-8b-8192",  # Example model name - this has been deprecated
        #model="llama-3.1-8b-instant",
        #model="canopylabs/orpheus-arabic-saudi", # speech model
        model="openai/gpt-oss-120b",

        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short poem about the ocean."}
        ],
        max_tokens=100
    )

    # Print the generated text
    print(response.choices[0].message.content)

except Exception as e:
    # Handle errors gracefully
    print(f"Error: {e}")


import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def analyze_attachment(attachment_url: str) -> str:
    messages = [
        {"role": "system", "content": "You are an AI that labels, summarizes, and extracts insights."},
        {
            "role": "user",
            "content": f"Analyze the content at {attachment_url} and return a JSON of labels, summary, and key insights."
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response["choices"][0]["message"]["content"]

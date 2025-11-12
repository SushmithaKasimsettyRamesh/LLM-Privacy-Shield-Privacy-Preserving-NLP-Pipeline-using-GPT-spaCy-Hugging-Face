from openai import OpenAI
from dotenv import load_dotenv
import os

# 1️⃣ Load environment variables from .env
load_dotenv()

# 2️⃣ Get the key (make sure it's set)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file.")

# 3️⃣ Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 4️⃣ Simple test request
print("🔍 Sending test request to OpenAI...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Just say 'Hello from LLM Privacy Shield!'"}
    ]
)

print("✅ Response:", response.choices[0].message.content)

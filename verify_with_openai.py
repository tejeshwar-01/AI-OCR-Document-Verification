import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load keys and endpoints from your .env file
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"  # use the version from your Azure portal if different
)

# Path to the processed data from Module 1
AZURE_JSON_PATH = r"C:\Users\tejes\OneDrive\Documents\Desktop\AIOCR\processed_output\azure_formatted_data.json"

# Load extracted OCR data
with open(AZURE_JSON_PATH, "r") as f:
    data = json.load(f)

# Pick one record to verify (for example, first Aadhaar document)
sample = data[0] if isinstance(data, list) else data

prompt = f"""
You are an AI document verification assistant.
Verify if the following extracted Aadhaar data is valid and consistent:

{json.dumps(sample, indent=2)}

Explain any issues found, such as missing fields, mismatched values, or invalid format.
"""

# Send the request to Azure OpenAI
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[{"role": "user", "content": prompt}],
    max_tokens=400
)

# Extract and print response text
result = response.choices[0].message.content
print("\n✅ Verification Result:\n")
print(result)

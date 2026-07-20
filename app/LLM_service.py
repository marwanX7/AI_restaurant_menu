import os
import json
from openai import AsyncOpenAI
from app.schemas import RecipeCreate


PROVIDERS = {
    "groq": {
        "client": AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        ),
        "model": "llama-3.3-70b-versatile",
    },
    "hf": {
        "client": AsyncOpenAI(
            api_key=os.getenv("HF_TOKEN"),
            base_url="https://router.huggingface.co/v1",
        ),
        "model": "openai/gpt-oss-120b:cerebras",
    },
}


async def extract_recipe(text: str, provider: str = "groq"):

    provider_config = PROVIDERS.get(provider)

    if not provider_config:
        raise ValueError("Invalid provider")

    client = provider_config["client"]
    model = provider_config["model"]

    messages = [
        {
            "role": "system",
            "content": """
You are an expert recipe information extractor.

Extract recipe information from the user's text.

Return ONLY valid JSON.
Do not return markdown.
Do not explain anything.

Return exactly these fields:
- name
- category
- ingredients
- instructions
- preparation_time
- calories
- price

Rules:
- Category MUST be exactly one of:
  Pizza
  Burger
  Pasta
  Dessert
  Drink
- If the recipe is a pizza, return "Pizza".
- If the recipe is a burger, return "Burger".
- If the recipe is a pasta dish, return "Pasta".
- If the recipe is a dessert, return "Dessert".
- If the recipe is a drink, return "Drink".
- Never return values like "Italian" or "Italian pizza".
- Ingredients MUST be returned as an array of strings.
- Preparation time must be an integer (minutes).
- Calories must be an integer.
- Price must be a number.
- Use null for any missing field.
- Return valid JSON only.
"""
        },
        {
            "role": "user",
            "content": text
        }
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    return RecipeCreate(**data)
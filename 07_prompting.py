"""
07_prompting.py
=================
Builds the final grounded prompt from a query + context package, and
sends it to an LLM via the OpenRouter API.

API key handling (per project rules):
  - Never hardcode a real key here.
  - Locally: set the OPENROUTER_API_KEY environment variable.
  - On Streamlit Cloud: leave it unset here; streamlit_app.py fills it in
    from st.secrets at runtime (see streamlit_app.py).
"""

import os

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(query: str, context_text: str) -> str:
    """
    A grounded, source-citing prompt. Deliberately strict: the model is
    told to say so rather than guess when context is missing, and to
    prefer current sources over outdated ones when both appear.
    """
    return f"""You are a personal finance assistant. Answer using ONLY the context below.

Rules:
1. Do not use outside knowledge beyond what is in the context.
2. If the context does not contain enough information to answer, say so
   clearly instead of guessing.
3. If a source is marked OUTDATED, do not use it as the basis for your
   answer — mention it only to note that the information has changed.
4. If sources conflict, state the conflict and prefer the CURRENT source.
5. Always cite the source numbers you used, e.g. (Source 1).

Context:
{context_text}

Question:
{query}

Answer:"""


def generate_answer(query: str, context_text: str, api_key: str = None, model: str = None) -> str:
    """Send the grounded prompt to the LLM via OpenRouter and return the answer text."""
    key = api_key or OPENROUTER_API_KEY
    model_name = model or OPENROUTER_MODEL

    if not key:
        return ("[No API key configured. Set OPENROUTER_API_KEY as an environment "
                "variable locally, or as a Streamlit secret when deployed.]")

    prompt = build_prompt(query, context_text)

    response = requests.post(
     url=OPENROUTER_URL,
     headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
     },
     json={
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    },
    timeout=30,
)

if not response.ok:
    raise Exception(
        f"Status: {response.status_code}\n"
        f"Response: {response.text}"
    )

data = response.json()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    sample_context = (
        "[Source 1] Making a Budget | 2023-03-13 | CURRENT\n"
        "Begin by recording every source of income you receive..."
    )
    print(build_prompt("How do I create a budget?", sample_context))
    print("\n(Set OPENROUTER_API_KEY to actually call the model.)")

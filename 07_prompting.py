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


def build_plan_prompt(query: str, summary: dict, context_text: str) -> str:
    """
    A grounded plan-building prompt. Combines the user's ACTUAL tracked
    financial numbers (from the local expense tracker) with the general
    knowledge context (from the RAG pipeline), so the plan is both
    personalized and grounded in sound guidance rather than invented.
    """
    category_lines = "\n".join(
        f"  - {row['category']}: {row['total']:.0f}" for row in summary["by_category"]
    ) or "  (لا توجد مصاريف مسجلة هذا الشهر بعد)"

    financial_snapshot = (
        f"Monthly income recorded: {summary['total_income']:.0f}\n"
        f"Monthly expenses recorded: {summary['total_expenses']:.0f}\n"
        f"Current balance: {summary['balance']:.0f}\n"
        f"Expenses by category:\n{category_lines}"
    )

    return f"""You are a personal finance assistant building a savings/budget plan for one specific user.

Use BOTH of the following inputs:
1. The user's ACTUAL financial snapshot below — these are real recorded numbers. Use them directly; never invent numbers that aren't given.
2. The general knowledge context below — grounded best-practice guidance. Cite a source when you use it, e.g. (Source 1).

User's financial snapshot (current month):
{financial_snapshot}

General knowledge context:
{context_text}

User's goal, in their own words:
{query}

Instructions:
- Write a short, concrete, numbered action plan (4-6 steps) tailored to this user's actual numbers and stated goal.
- If the snapshot is missing key data (e.g. no income recorded, or too few expenses tracked to see a pattern), say so plainly and ask what's missing instead of guessing.
- Reply in the same language the user's goal was written in.
- End with one clear next action the user can take today.

Plan:"""


def generate_plan(query: str, summary: dict, context_text: str,
                   api_key: str = None, model: str = None) -> str:
    """Send the grounded plan prompt to the LLM via OpenRouter and return the plan text."""
    key = api_key or OPENROUTER_API_KEY
    model_name = model or OPENROUTER_MODEL

    if not key:
        return ("[No API key configured. Set OPENROUTER_API_KEY as an environment "
                "variable locally, or as a Streamlit secret when deployed.]")

    prompt = build_plan_prompt(query, summary, context_text)

    response = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 700,
        },
        timeout=30,
    )

    if not response.ok:
        raise Exception(
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


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
            "max_tokens": 700,
        },
        timeout=30,
    )

    if not response.ok:
        raise Exception(
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    sample_context = (
        "[Source 1] Making a Budget | 2023-03-13 | CURRENT\n"
        "Begin by recording every source of income you receive..."
    )
    print(build_prompt("How do I create a budget?", sample_context))
    print("\n(Set OPENROUTER_API_KEY to actually call the model.)")

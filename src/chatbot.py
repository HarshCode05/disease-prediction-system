"""
chatbot.py
-----------
Handles the follow-up conversation layer using an LLM API.

Uses Google Gemini (free tier, no credit card needed) via its
OpenAI-compatible endpoint - this means we can keep using the same
`openai` Python library and message format, just pointed at Google's
servers instead of OpenAI's, with a Gemini model name.

This is called AFTER a prediction is made, to give the user
general, non-diagnostic lifestyle guidance based on their result.

IMPORTANT: This is not a medical professional replacement. The
system prompt below explicitly constrains the assistant to general
wellness information and tells it to recommend seeing a doctor for
anything specific - this is a responsible-AI guardrail, not optional.
"""

import os
from openai import OpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

# The client is created lazily (inside functions, not at import time) so that
# a missing GEMINI_API_KEY doesn't crash the entire Streamlit app on startup -
# it only raises an error when the chatbot is actually used, and the app
# handles that gracefully (see streamlit_app.py's try/except around chatbot calls).

def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before using the chatbot feature."
        )
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


SYSTEM_PROMPT = """You are a friendly, cautious health-information assistant embedded in a
disease-risk-prediction app (Heart Disease, Diabetes, and Chronic Kidney Disease).

Your role:
- Explain, in plain language, what a risk prediction result generally means.
- Offer general, well-established lifestyle information (diet, exercise, sleep, stress)
  relevant to the disease in question.
- Encourage the user to consult a licensed doctor for diagnosis, testing, or treatment
  decisions - you are not a substitute for professional medical care.

Strict rules:
- Never claim to diagnose the user or confirm/deny they have a disease.
- Never suggest specific medication names, dosages, or treatment plans.
- If the user describes symptoms suggesting a medical emergency, tell them to seek
  immediate in-person or emergency medical care.
- Keep responses concise and easy to understand for a non-medical audience.
"""


def get_followup_response(disease: str, risk_result: str, risk_probability: float, user_message: str) -> str:
    """
    Get a chatbot response for a follow-up question after a prediction.

    Args:
        disease: one of "Heart Disease", "Diabetes", "Chronic Kidney Disease"
        risk_result: "High Risk" or "Low Risk"
        risk_probability: model's predicted probability (0-1)
        user_message: the user's question/message to the chatbot

    Returns:
        The assistant's reply as a string.
    """
    context_message = (
        f"The user was just assessed for {disease}. "
        f"Model result: {risk_result} (predicted probability: {risk_probability:.0%}). "
        f"Now respond to their message: {user_message}"
    )

    response = _get_client().chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_message},
        ],
        max_tokens=400,
        temperature=0.5,
    )

    return response.choices[0].message.content


def get_initial_message(disease: str, risk_result: str, risk_probability: float) -> str:
    """
    Generate the chatbot's opening message right after a prediction is shown,
    so the conversation doesn't start on a blank screen.
    """
    prompt = (
        f"The user just received a {risk_result} result for {disease} "
        f"(predicted probability: {risk_probability:.0%}). "
        f"Write a short (2-3 sentence) opening message acknowledging the result "
        f"and inviting them to ask questions about lifestyle or next steps."
    )

    response = _get_client().chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.5,
    )

    return response.choices[0].message.content

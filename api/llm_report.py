"""
Generates a human-readable, cautious explanation report from the
model's structured output (prediction + confidence + Grad-CAM focus
region). Uses the Groq API (free tier, OpenAI-compatible, no billing
setup required).

IMPORTANT: The LLM is explicitly instructed to describe/explain the
model's output, NOT to make an independent diagnosis. This keeps the
system safe and keeps the LLM's role scoped correctly.
"""

import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are a medical-imaging AI assistant that writes clear, \
cautious explanations of a deep learning model's output for a chest X-ray \
classifier. You are NOT a doctor and you must NEVER present the output as a \
confirmed diagnosis.

Rules:
- Describe what the MODEL predicted and how confident it is, in plain language.
- Reference the region of the image the model focused on (from Grad-CAM), \
described in general terms.
- Explicitly state this is an AI-assisted screening aid, not a medical diagnosis.
- Recommend the patient consult a licensed radiologist/physician for confirmation.
- Keep the tone calm, clear, and non-alarming.
- Length: 4-7 sentences.
"""


def generate_report(predicted_class: str, confidence: float, focus_region: str) -> str:
    user_prompt = f"""
Model prediction: {predicted_class}
Model confidence: {confidence * 100:.1f}%
Grad-CAM focus region: {focus_region}

Write the explanation report now.
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"[LLM report unavailable: {e}] "
            f"The model predicted '{predicted_class}' with {confidence*100:.1f}% confidence, "
            f"focusing on the {focus_region}. This is an AI-assisted screening result, "
            f"not a medical diagnosis -- please consult a licensed physician."
        )
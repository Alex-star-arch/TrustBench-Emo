"""Prompt templates and output parsers for TrustBench-Emo."""
import re

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

INSTRUCTION = (
    "Classify the emotion. "
    "Reply with exactly one label from {angry, disgust, fear, happy, neutral, sad, surprise} "
    "and a one-sentence reason."
)


def image_prompt():
    """Return the text prompt used for image inputs (FER2013)."""
    return INSTRUCTION


def text_prompt(utterance):
    """Return the full text prompt used for text inputs (GoEmotions)."""
    return f"Text: {utterance}\n{INSTRUCTION}"


def parse_output(text):
    """Extract (label, rationale) from model free-text output.

    Strategy: find the first occurrence of any allowed emotion label, then
    treat the remaining sentence after the label as the rationale.
    """
    text = text.strip()
    label = None
    lower = text.lower()
    for emo in EMOTIONS:
        if emo in lower:
            label = emo
            break
    if label is None:
        return None, text
    # remove label token from text and clean up rationale
    rationale = re.sub(r"(?i)" + re.escape(label), "", text, count=1)
    rationale = re.sub(r"^[^a-zA-Z]*", "", rationale)
    rationale = rationale.strip(" .,;:-")
    return label, rationale

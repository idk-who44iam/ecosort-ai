"""
EcoSort AI - image classifier.

Images are classified by SmolVLM2-2.2B through Ollama. If Ollama or the model
is unavailable the call raises an error instead of substituting a different
classifier.

Configure with environment variables:
    OLLAMA_URL           e.g. http://127.0.0.1:11434
    CLASSIFIER_BACKEND   auto | qwen   (default: auto)
"""
import base64
import io
import json
import os
import re
import urllib.request

import numpy as np
import pillow_avif  # Registers AVIF support with Pillow.
from PIL import Image

# Material categories returned by the vision model and used by the training tool
CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

ADVICE = {
    "paper": (
        "Paper is highly recyclable. Keep it dry and free of food residue, "
        "remove plastic windows from envelopes, and place it in the paper "
        "recycling bin."
    ),
    "plastic": (
        "Check the recycling number on the item (1, 2 and 5 are usually "
        "widely recycled). Rinse it out and put it in the plastics bin. "
        "Lids should be left on empty bottles."
    ),
    "glass": (
        "Glass is 100% recyclable and can be recycled endlessly. Rinse the "
        "container, remove the lid, and place it in the glass recycling "
        "bin. Do NOT put broken glass, ceramics or light bulbs in with it."
    ),
    "metal": (
        "Cans and metal tins are infinitely recyclable. Rinse them out and "
        "place them loose (not in bags) in the metal recycling bin."
    ),
    "cardboard": (
        "Flatten cardboard boxes before recycling so they take up less "
        "space. Keep them dry and remove any tape or plastic padding first."
    ),
    "trash": (
        "This item doesn't match a common recyclable. Place it in the "
        "general waste bin, or check your local recycling guidelines — "
        "some items can't go in mixed recycling."
    ),
}

VERDICT_LABELS = {
    "recyclable": "Recyclable",
    "not_recyclable": "Not recyclable",
    "depends": "Depends",
}

VERDICT_GUIDANCE = {
    "recyclable": "This material is commonly recyclable. Rinse or clean it and check local collection rules.",
    "not_recyclable": "This is classified as general waste rather than a commonly recycled material.",
    "depends": "Check the recycling label or number and your local collection rules. Plastic acceptance varies by type.",
}

ICONS = {
    "paper": "📄",
    "plastic": "🧴",
    "glass": "🍾",
    "metal": "🥫",
    "cardboard": "📦",
    "trash": "🗑️",
}

# The exact feature length must match the training script.
IMG_SIZE = 96
HIST_BINS = 16
FEATURE_LEN = HIST_BINS * 3 + 4  # hue + saturation + value histograms + 4 stats


def extract_features(image):
    """
    Turn a PIL image into a fixed-length feature vector.

    Features used:
      * colour histogram (Hue, Saturation, Value channels)
      * brightness / saturation statistics
      * edge density (how 'textured' the surface looks)
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    hsv = np.array(img.convert("HSV"))

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    feats = []
    for channel in (hue, sat, val):
        hist, _ = np.histogram(channel, bins=HIST_BINS, range=(0, 256))
        feats.append(hist / (hist.sum() + 1e-6))

    # Greyscale texture stats
    grey = np.array(img.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(grey, axis=1))
    gy = np.abs(np.diff(grey, axis=0))
    grad = gx[:-1, :] + gy[:, :-1]  # both (IMG_SIZE-1, IMG_SIZE-1)
    feats.append(
        [
            float(grey.std()),
            float(sat.mean() / 255.0),
            float(val.mean() / 255.0),
            float((grad > 30).mean()),
        ]
    )

    return np.concatenate(feats).astype(np.float32)


# --------------------------------------------------------------------------
# Vision backend (local or remote Ollama server)
# --------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
BACKEND = os.environ.get("CLASSIFIER_BACKEND", "auto").lower()


# --------------------------------------------------------------------------
# The app's fixed vision model
# --------------------------------------------------------------------------
BUILTIN_MODELS = [
    {
        "id": "smolvlm2-2.2b",
        "tag": "ahmadwaqar/smolvlm2-2.2b-instruct",
        "label": "SmolVLM2 2.2B",
        "params": "2.2B",
        "description": "The app's fixed vision model.",
        "max_side": 512,
        "timeout": 180,
    },
]


def normalise_tag(tag):
    """Ollama reports untagged models with an implicit ':latest' suffix."""
    return str(tag).strip().lower().removesuffix(":latest")


VISION_MODELS = [dict(model) for model in BUILTIN_MODELS]


def default_model():
    """Return the single model used for every image."""
    return dict(VISION_MODELS[0])


def get_model(key):
    """Look a model up by id, tag or label. Returns None when unknown."""
    if not key:
        return None
    key = str(key).strip().lower()
    for m in VISION_MODELS:
        if key in (m["id"].lower(), m["tag"].lower(), m["label"].lower()):
            return m
    return None


def public_models():
    """Model definitions that are safe to render in the browser."""
    fields = ("id", "tag", "label", "params", "description")
    return [{k: m[k] for k in fields} for m in VISION_MODELS]


def list_installed_models(timeout=4):
    """
    Ask the Ollama server which models it actually has.

    Returns a set of normalised model names, or None if the server is unreachable.
    """
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 - offline, bad URL, not JSON...
        return None
    names = {normalise_tag(m.get("name", "")) for m in data.get("models", [])}
    return {n for n in names if n}

_Q_PROMPT = (
    "Look at the image. Select the main object physical material only, not "
    "whether it is recyclable. Clear plastic bottles/cups are plastic, never "
    "glass. A crumpled sheet of paper is paper; cardboard is a thick box/board. "
    "Choose exactly one category: cardboard, glass, metal, paper, plastic, "
    "trash. Do not guess missing labels. Return only the required JSON."
)

_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        }
    },
    "required": ["category"],
}


def _downscale_to_base64(image_path, max_side=768, quality=85):
    """Downscale the photo and return it as a base64 JPEG for the API."""
    with Image.open(image_path) as img:
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def validate_image(image_path):
    """Reject unreadable or unsupported uploads with an actionable message."""
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(
            "This image could not be decoded. Please upload a valid JPG, PNG, "
            "WEBP, BMP, GIF, or AVIF image."
        ) from exc


def _ask_qwen(image_path):
    """Send the photo to SmolVLM2 and return the raw reply."""
    model = default_model()
    payload = {
        "model": model["tag"],
        "stream": False,
        "format": _CATEGORY_SCHEMA,
        "keep_alive": "2h",
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 24,
        },
        "messages": [
            {
                "role": "user",
                "content": _Q_PROMPT,
                "images": [_downscale_to_base64(image_path, max_side=model["max_side"])],
            }
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=model["timeout"]) as resp:
        data = json.loads(resp.read().decode())
    return data["message"]["content"]


def _parse_response(text):
    """Extract and validate the model's structured material classification."""
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("The vision model did not return a JSON object.")
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(parsed, dict):
        raise ValueError("The vision model returned an invalid classification.")

    category = str(parsed.get("category", "")).strip().lower()
    if category not in CATEGORIES:
        raise ValueError(f"The vision model returned an unknown category: {category!r}")
    return category


def predict_qwen(image_path):
    """Return the material category selected by SmolVLM2."""
    return _parse_response(_ask_qwen(image_path))


def recycling_verdict(category):
    """Apply consistent household recycling guidance to the detected material."""
    if category == "plastic":
        return "depends"
    if category == "trash":
        return "not_recyclable"
    if category in CATEGORIES:
        return "recyclable"
    raise ValueError(f"Unknown material category: {category!r}")


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _probe(model, timeout=6):
    """
    Quick health check: is the server up and does it have this model?

    Without this, an offline server would block the upload for the full
    per-model timeout before reporting that it is unavailable.
    Returns (ready, problem_description).
    """
    installed = list_installed_models(timeout=timeout)
    if installed is None:
        return False, f"the AI server at {OLLAMA_URL} is unreachable"
    if normalise_tag(model["tag"]) not in installed:
        return False, f"model '{model['tag']}' is not installed on the AI server"
    return True, ""


def predict(image_path):
    """
    Classify an image file using the app's fixed vision model.

    Returns a dict with material category, recycling verdict, model details and warning.

    Raises RuntimeError when the server or model is unavailable, so the caller
    can show a real error instead of a misleading guess.
    """
    chosen = default_model()
    if BACKEND not in ("auto", "qwen"):
        raise RuntimeError(
            "CLASSIFIER_BACKEND must be 'auto' or 'qwen' for vision-model "
            "analysis. Local Random Forest mode is no longer supported."
        )

    ready, problem = _probe(chosen)
    if not ready:
        raise RuntimeError(f"{chosen['label']} unavailable ({problem}).")

    try:
        category = predict_qwen(image_path)
    except Exception as exc:  # noqa: BLE001 - bad reply, timeout, or request failure
        raise RuntimeError(f"{chosen['label']} image analysis failed: {exc}") from exc

    return {
        "category": category,
        "verdict": recycling_verdict(category),
        "source": "vision",
        "model_id": chosen["id"],
        "model_tag": chosen["tag"],
        "model_label": chosen["label"],
        "warning": None,
    }

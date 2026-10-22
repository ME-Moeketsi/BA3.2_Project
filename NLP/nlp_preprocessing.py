"""
Predict-Care AI
NLP Preprocessing

Here I am cleaning patient chief-complaint text before TF-IDF
feature extraction.

The cleaner is deliberately conservative: it lowercases and
normalises whitespace, expands a small hand-curated set of
paraphrases that are clinically equivalent, and leaves the rest
of the wording intact. Anything that would destroy signal
(e.g. aggressive punctuation removal in the middle of clinically
meaningful tokens) is avoided.

The character-level n-gram vectorizer used in train_ktas_model.py
handles typos and near-miss spellings; this module's job is only
to reduce trivial variants so those n-grams line up.
"""

import re


# ============================================================
# SMALL HAND-CURATED PARAPHRASE MAP
# ============================================================

_PARAPHRASE_MAP = {
    "cardiac arrest": [
        "heart attack",
        "cardiac arrest",
        "cardiac arreset",
        "cardiac arest",
    ],
    "shortness of breath": [
        "shortness of breath",
        "short of breath",
        "breathlessness",
        "breath lessness",
        "dyspnea",
        "dyspnoea",
        "sob",
        "difficulty breathing",
        "labored breathing",
    ],
    "chest pain": [
        "chest pain",
        "chest tightness",
        "tight chest",
        "chest pressure",
        "pain in chest",
        "cp",
    ],
    "abdominal pain": [
        "abdominal pain",
        "belly pain",
        "stomach pain",
        "stomach ache",
        "stomache ache",
        "tummy ache",
        "abd pain",
        "abd",
    ],
    "vomiting": [
        "vomiting",
        "vomit",
        "throwing up",
        "puke",
        "puking",
    ],
    "fever": [
        "fever",
        "high temperature",
        "temperature",
        "pyrexia",
    ],
    "headache": [
        "headache",
        "head ache",
        "head pain",
        "migraine",
    ],
    "back pain": [
        "back pain",
        "backache",
        "back ache",
        "lbp",
        "low back pain",
    ],
    "dizziness": [
        "dizziness",
        "dizzy",
        "lightheaded",
        "light headed",
        "light headedness",
        "vertigo",
    ],
    "unconsciousness": [
        "unconsciousness",
        "unconscious",
        "unresponsive",
        "passed out",
        "fainted",
    ],
    "seizure": [
        "seizure",
        "convulsion",
        "convulsions",
        "fits",
    ],
    "bleeding": [
        "bleeding",
        "severe bleeding",
        "hemorrhage",
        "haemorrhage",
    ],
    "laceration": [
        "laceration",
        "cut",
        "cuts",
        "gash",
    ],
    "burn": [
        "burn",
        "burns",
        "burned",
        "burnt",
    ],
}


_SORTED_PARAPHRASE_KEYS = sorted(
    _PARAPHRASE_MAP.keys(), key=len, reverse=True
)

_PARAPHRASE_PATTERNS = [
    (
        re.compile(r"\b" + re.escape(key) + r"\b"),
        _PARAPHRASE_MAP[key],
    )
    for key in _SORTED_PARAPHRASE_KEYS
]


# ============================================================
# PUBLIC CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Here I am normalising chief-complaint text.

    Steps, in order:
      1. Coerce to string, lowercase, strip surrounding whitespace.
      2. Replace non-alphanumeric characters with spaces.
      3. Collapse repeated whitespace.
      4. Expand any phrase variant that matches a canonical
         paraphrases entry.
    """

    if text is None:
        return ""

    cleaned = str(text).lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    for _ in range(2):
        previous = cleaned
        cleaned = _apply_paraphrase_expansions(cleaned)
        if cleaned == previous:
            break

    return cleaned


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _text_contains_phrase(haystack, phrase):
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, haystack) is not None


def _apply_paraphrase_expansions(text):
    all_variants = []
    for canonical, variants in _PARAPHRASE_MAP.items():
        for variant in variants:
            all_variants.append((variant, canonical))

    all_variants.sort(key=lambda pair: len(pair[0]), reverse=True)

    result = text
    for variant, canonical in all_variants:
        if variant == canonical:
            continue
        pattern = r"\b" + re.escape(variant) + r"\b"
        if re.search(pattern, result):
            result = re.sub(pattern, canonical, result)

    return result


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    examples = [
        "Chest Pain",
        "chest-pain",
        "CHEST PAIN",
        "chest tightness",
        "arm pain, Lt",
        "LBP - Low back pain",
        "RUQ pain",
        "Open Wound",
        "heart attack",
        "short of breath",
        "puking",
        "high temperature",
        "stomache ache",
        "unconscious",
        "MISSING",
    ]

    print("NLP PREPROCESSING TEST")
    print("----------------------")

    for example in examples:
        print(f"{example!r} -> {clean_text(example)!r}")
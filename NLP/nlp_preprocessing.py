import re


def clean_text(text):
    """
    Clean patient chief complaint text before NLP processing.
    """

    # Convert text to lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----- TEST TEXT PREPROCESSING -----

if __name__ == "__main__":

    examples = [
        "Chest Pain",
        "arm pain, Lt",
        "LBP - Low back pain",
        "RUQ pain",
        "Open Wound"
    ]

    print("NLP PREPROCESSING TEST")
    print("----------------------")

    for example in examples:
        print(f"{example} -> {clean_text(example)}")
import re
import unicodedata


def preprocess(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-\"<>\[\]()/#&:%]", "", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(\s*\)\s*", " ", text)
    text = re.sub(r"\s@\s*", ' ', text)
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)
    text = re.sub(r'([!?.,;:\"\'])\1+', r'\1', text)
    text = re.sub(r"([A-Za-z])\1{2,}", r"\1", text)
    text = re.sub(r"([A-Za-z])\1{1}", r"\1\1", text)
    time_pattern = re.compile(
        r"""
        (?:
            \b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b
            | \b\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}\b
            | \b\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*\d{2,4}\b
            | \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{1,2},?\s*\d{2,4}\b
        )
        (?:\s*\(?(?:UTC|GMT|PST|EST|CET|IST)\)?)?
        | \b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|UTC|GMT)?\b
        """,
        flags=re.IGNORECASE | re.VERBOSE
    )
    text = re.sub(time_pattern, "<DATE>", text)
    text = re.sub(r"\"\s+", "", text)
    return text.strip()
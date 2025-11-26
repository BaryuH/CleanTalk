import os
from dotenv import load_dotenv
import json
import re
import unicodedata
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT_ZEROSHOT = """
You are a strict toxic comment classifier for the “Toxic Comment Classification Challenge” (Jigsaw/Google on Kaggle).

Given a single English comment, you must decide for each of the following labels whether it applies (1) or not (0):

- toxic: rude, disrespectful, or unreasonable comments likely to make someone leave a conversation. Includes general profanity, harassment, or bullying.
- severe_toxic: a severe, extreme form of toxicity. Includes wishes of harm, intense harassment, or slurs combined with threats or strong abuse. This is a subset of toxic.
- obscene: contains strong profanity, vulgar or sexually explicit terms, or highly offensive language.
- threat: expresses an intention to inflict physical harm, death, or serious damage to a person, group, or property. Includes conditional or indirect threats.
- insult: negative or demeaning comments about a person, their characteristics, or behaviour. Includes name-calling and personal attacks but not necessarily profanity.
- identity_hate: hateful or abusive content targeting a protected group or identity (e.g. race, ethnicity, nationality, religion, gender, sexual orientation, disability, etc.), using slurs or dehumanizing language.

Rules:
- Multiple labels can be 1 at the same time.
- If the comment is clean or neutral, all labels must be 0.
- “severe_toxic” MUST be 1 only when “toxic” is also clearly 1.
- Focus on the literal content of the comment, not possible missing context.
- Be conservative and follow the Jigsaw definitions above.

Output format:
- You MUST respond with a single valid JSON object.
- The JSON MUST have exactly these keys and no others:
  toxic, severe_toxic, obscene, threat, insult, identity_hate
- Each value MUST be either 0 or 1 (integers, not booleans, not strings).
- Do not output any explanation, text, or formatting outside the JSON.
"""

SYSTEM_PROMPT_FEWSHOT = """
You are a strict toxic comment classifier for the “Toxic Comment Classification Challenge” (Jigsaw/Google on Kaggle).

Task:
Given a single English comment, decide for each of the following labels whether it applies (1) or not (0):

- toxic: rude, disrespectful, or unreasonable comments likely to make someone leave a conversation. Includes general profanity, harassment, or bullying.
- severe_toxic: a severe, extreme form of toxicity. Includes wishes of harm, intense harassment, or slurs combined with threats or strong abuse. This is a subset of toxic.
- obscene: contains strong profanity, vulgar or sexually explicit terms, or highly offensive language.
- threat: expresses an intention to inflict physical harm, death, or serious damage to a person, group, or property. Includes conditional or indirect threats.
- insult: negative or demeaning comments about a person, their characteristics, or behaviour. Includes name-calling and personal attacks.
- identity_hate: hateful or abusive content targeting a protected group or identity (e.g. race, ethnicity, nationality, religion, gender, sexual orientation, disability, etc.), using slurs or dehumanizing language.

Labeling rules:
- Multiple labels can be 1.
- If the comment is clean or neutral, all labels must be 0.
- “severe_toxic” = 1 only for very strong / extreme toxicity, and only when “toxic” = 1.
- Focus strictly on the literal content of the comment.

Output format (very important):
- Always output a single valid JSON object with EXACTLY these keys:
  toxic, severe_toxic, obscene, threat, insult, identity_hate
- Each value must be 0 or 1 (integers).
- No extra keys, no comments, no code blocks, no markdown, no explanations.

Here are some examples:

Example 1
Comment:
"fuck your filthy mother in the ass, dry!"
Output:
{"toxic":1,"severe_toxic":0,"obscene":1,"threat":0,"insult":1,"identity_hate":0}

Example 2
Comment:
"tony sidaway is obviously a fistfuckee. he loves an arm up his ass."
Output:
{"toxic":1,"severe_toxic":0,"obscene":1,"threat":0,"insult":1,"identity_hate":0}

Example 3
Comment:
"they are not original research, they are pointed in the episodes."
Output:
{"toxic":0,"severe_toxic":0,"obscene":0,"threat":0,"insult":0,"identity_hate":0}

Example 4
Comment:
"hi! i am back again! last warning! stop undoing my edits or die!"
Output:
{"toxic":1,"severe_toxic":0,"obscene":0,"threat":1,"insult":0,"identity_hate":0}

Example 5
Comment:
"you are a stupid fuck and your mother's cunt stinks"
Output:
{"toxic":1,"severe_toxic":1,"obscene":1,"threat":0,"insult":1,"identity_hate":0}

Now classify the next comment following exactly the same JSON format as in the examples.
"""

def classify_comment_zero_shot(comment: str, model: str = "gpt-5") -> dict:
    """
    Zero-shot toxic comment classification (multi-label, Jigsaw style).
    Returns a dict with 6 keys, each ∈ {0, 1}.
    """

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT_ZEROSHOT,
        input=f'Comment:\n"{comment}"\n\nClassify this comment according to the task.'
    )

    raw = response.output_text.strip()

    try:
        labels = json.loads(raw)
    except Exception:
        print("Model output not valid JSON:", raw)
        raise

    expected_keys = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    cleaned = {}
    for k in expected_keys:
        v = labels.get(k, 0)
        cleaned[k] = 1 if str(v) in ["1", "true", "True"] else 0

    return cleaned

def classify_comment_few_shot(comment: str, model: str = "gpt-5") -> dict:
    """
    Zero-shot toxic comment classification (multi-label, Jigsaw style).
    Returns a dict with 6 keys, each ∈ {0, 1}.
    """

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT_FEWSHOT,
        input=f'Comment:\n"{comment}"\n\nClassify this comment according to the task.'
    )

    raw = response.output_text.strip()

    try:
        labels = json.loads(raw)
    except Exception:
        print("Model output not valid JSON:", raw)
        raise

    expected_keys = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    cleaned = {}
    for k in expected_keys:
        v = labels.get(k, 0)
        cleaned[k] = 1 if str(v) in ["1", "true", "True"] else 0

    return cleaned


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


if __name__ == "__main__":
    print("--- Start the program ---")

    while True:
        test_comment = input("Enter comment: ")

        if test_comment == "quit":
            break

        options = int(input("Choose zeroshot=0, fewshot=1, both=2: "))

        test_comment = preprocess(test_comment)

        if options == 0:
            result = classify_comment_zero_shot(test_comment)
            print(f"Comment: {test_comment}\n")
            print(f"Labels: {result}\n")
        elif options == 1:
            result = classify_comment_few_shot(test_comment)
            print(f"Comment: {test_comment}\n")
            print(f"Labels: {result}\n")
        elif options == 2:
            print(f"Comment: {test_comment}\n")

            result = classify_comment_zero_shot(test_comment)
            print(f"Labels (zeroshot): {result}\n")

            result = classify_comment_few_shot(test_comment)
            print(f"Labels (fewshot): {result}\n")
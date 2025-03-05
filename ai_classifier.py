import os
import openai
import pandas as pd
from dotenv import load_dotenv

# 1. Load environment variables from .env (or any other secure location)
load_dotenv()  # This will read OPENAI_API_KEY if present in .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def classify_comment(comment_body: str) -> int:
    """
    Sends a prompt to the GPT model to classify text as 1 (Hate Speech) or 0 (Not Hate Speech).
    Returns 1 if 'Hate Speech', else 0.
    """
    # System prompt with definition of hate speech in Lithuanian context:
    system_prompt = (
        "You are a helpful assistant specialized in detecting hate speech in Lithuanian text from reddit comments. "
        "Hate speech is defined as any text that:\n"
        "• Expresses serious hatred or violence towards a person or group based on attributes such as "
        "  race, ethnicity, religion, gender, sexual orientation, or disability.\n"
        "• Uses extremely offensive, demeaning, or dehumanizing language targeted at a protected group.\n\n"
        "You will be provided with a piece of text. Classify it strictly as either:\n"
        "'Hate Speech' or 'Not Hate Speech'.\n\n"
        "If it includes hateful insults, incitement of violence, or strong discriminatory content, classify as 'Hate Speech'.\n"
        "Otherwise, classify as 'Not Hate Speech'.\n\n"
        "Important: If you are uncertain, default to 'Not Hate Speech'.\n"
    )

    # The actual user prompt
    user_prompt = (
        f"Tekstas:\n"
        f"{comment_body}\n\n"
        f"Atsakyk tik vienu žodžiu: 'Hate Speech' arba 'Not Hate Speech'."
    )

    try:
        print("[DEBUG] Sending request to OpenAI ChatCompletion...")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo", etc. (Replace with your valid model)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # Low temperature => more consistent
        )
        classification = response.choices[0].message.content.strip()
        print(f"[DEBUG] OpenAI response: '{classification}'")

        cleaned = classification.lower()

        # Return 1 if exactly "hate speech", else 0
        if cleaned == "hate speech":
            return 1
        else:
            return 0

    except Exception as e:
        # If there's an error (rate limit, network, etc.), print it and return fallback
        print(f"[ERROR] OpenAI API error: {e}")
        return 0  # Default to "Not Hate Speech" => 0


def main():
    print("[INFO] Loading CSV: 'datasets/v2/lietuvos_comments.csv' ...")
    df = pd.read_csv("datasets/v2/lietuvos_comments.csv")
    print(f"[INFO] Successfully loaded {len(df)} rows.")

    # Limit for testing so we don't use up tokens on large data
    limit = 5000 # Adjust or remove once you're ready
    nrows = min(limit, len(df))
    print(f"[INFO] Will process up to {nrows} rows.")

    # Classify each comment in the first nrows
    for i in range(nrows):
        comment_text = df.loc[i, "body"]
        # Show a truncated version of comment to avoid printing huge text
        truncated_text = (comment_text[:80] + "...") if len(comment_text) > 80 else comment_text
        print(f"\n[INFO] Processing row {i+1}/{nrows}")
        print(f"[INFO] Comment body (truncated): '{truncated_text}'")

        label = classify_comment(comment_text)
        print(f"[INFO] => is_hate_speech: {label}")
        df.loc[i, "is_hate_speech"] = label

    print("\n[INFO] Saving updated DataFrame to 'reddit_data_annotated.csv' ...")
    df.to_csv("/datasets/v2/reddit_data_annotated.csv", index=False)
    print(f"[INFO] Annotation complete for {nrows} rows. Results saved to 'reddit_data_annotated.csv'.")


if __name__ == "__main__":
    main()

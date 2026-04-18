import string
from itertools import combinations

def skip_bigrams(text: str) -> list[tuple[str]]:
    """
    Cleans the input text and returns all skip-bigrams.

    Parameters
    ----------
    text: str
        The input text in string format.

    Returns
    ----------
    result: list[tuple[str]]
        A list containing all the skip-bigrams from the input text.

    """
    tokens = text.split()

    cleaned_tokens = [
        token.strip(string.punctuation).lower()
        for token in tokens
        if token.strip(string.punctuation)  # Avoid empty tokens
    ]

    result = list(combinations(cleaned_tokens, 2))

    return result

def rouge_s_bigram_scorer(original: str, summary: str) -> float:
    """
    Compute the ROUGE-S score for skip-bigrams.
    """
    original_text_skip_bigrams = skip_bigrams(original)
    summary_text_skip_bigrams = skip_bigrams(summary)

    overlaps = 0
    for bigram in original_text_skip_bigrams:
        if bigram in summary_text_skip_bigrams:
            overlaps += 1

    rouge_s = overlaps / len(original_text_skip_bigrams)

    return rouge_s

original_text = "Timothy read Ameer’s Substack and thought to himself: “wow, this Substack is great, I have subscribed! And I think that anyone else who reads it should subscribe, since it does such an amazing job of breaking down technical concepts in such an accessible and interesting way."
summary_text = "Timothy praised Ameer’s amazing explanations of technical concepts in his Substack, and recommended that readers should subscribe."

rouge_s = rouge_s_bigram_scorer(original_text, summary_text)
print(f"The ROUGE-S bigram score is {rouge_s*100:.2f}%.")
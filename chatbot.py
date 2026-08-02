"""
CodeAlpha - Task 2: FAQ Chatbot
--------------------------------
A simple retrieval-based chatbot that answers user questions by finding the
most similar question in a pre-collected FAQ dataset.

Pipeline:
    1. Collect FAQs        -> loaded from faqs.json
    2. Preprocess text     -> NLTK: lowercase, tokenize, remove stopwords/punct, lemmatize
    3. Match user question -> TF-IDF vectors + cosine similarity
    4. Display best answer -> printed to console (run via command line)

Usage:
    python chatbot.py
    (then type your question, or "quit" to exit)
"""

import json
import string
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. One-time NLTK data download (only downloads if not already present)
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name} ...")
            nltk.download(name, quiet=True)


ensure_nltk_data()

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Lowercase, tokenize, strip punctuation/stopwords, and lemmatize."""
    text = text.lower().translate(PUNCT_TABLE)
    tokens = word_tokenize(text)
    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in STOP_WORDS
    ]
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# 3. The FAQ Chatbot class
# ---------------------------------------------------------------------------
class FAQChatbot:
    def __init__(self, faq_path: str = "faqs.json", similarity_threshold: float = 0.25):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]
        self.similarity_threshold = similarity_threshold

        # Combine each question with its keyword variants (if present) so that
        # paraphrased user queries (e.g. "refund" vs "return") still match.
        combined_texts = [
            item["question"] + " " + item.get("keywords", "")
            for item in self.faqs
        ]
        self.processed_questions = [preprocess(t) for t in combined_texts]

        # Fit TF-IDF vectorizer on the combined FAQ corpus
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def get_response(self, user_query: str):
        """Return (best_answer, matched_question, similarity_score)."""
        processed_query = preprocess(user_query)

        if not processed_query.strip():
            return (
                "I'm sorry, I didn't understand that. Could you rephrase your question?",
                None,
                0.0,
            )

        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.similarity_threshold:
            return (
                "I'm sorry, I don't have an answer for that. "
                "Please try rephrasing, or contact support@example.com for help.",
                None,
                best_score,
            )

        return self.answers[best_idx], self.questions[best_idx], best_score


# ---------------------------------------------------------------------------
# 4. Command-line chat loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" FAQ Chatbot (type 'quit' or 'exit' to stop)")
    print("=" * 60)

    bot = FAQChatbot(faq_path="faqs.json")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Have a great day.")
            break

        if not user_input:
            continue

        answer, matched_question, score = bot.get_response(user_input)

        if matched_question:
            print(f"Bot: {answer}")
            print(f"     (matched FAQ: \"{matched_question}\" | similarity: {score:.2f})")
        else:
            print(f"Bot: {answer}")


if __name__ == "__main__":
    sys.exit(main())

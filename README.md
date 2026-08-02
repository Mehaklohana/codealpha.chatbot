# CodeAlpha_FAQChatbot

A retrieval-based chatbot that answers user questions by matching them
against a collected set of FAQs, using NLP preprocessing and cosine
similarity — built for the CodeAlpha AI Internship, Task 2.

## Features
- **Collect FAQs**: stored in `faqs.json` (question, answer, and keyword
  variants for better matching)
- **Preprocess text**: NLTK — lowercasing, tokenization, punctuation removal,
  stopword removal, and lemmatization
- **Match user questions**: TF-IDF vectorization + cosine similarity against
  the FAQ set
- **Display best answer**: console chatbot (`chatbot.py`) and an optional
  simple web chat UI (`app.py` + `templates/index.html`)

## Files
```
faq_chatbot/
├── faqs.json          # FAQ dataset (questions, answers, keywords)
├── chatbot.py          # Core chatbot logic (console version) — run this first
├── app.py              # Optional Flask web UI
├── templates/
│   └── index.html      # Chat UI page (used by app.py)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd faq_chatbot
pip install -r requirements.txt
```

The first run will automatically download the small NLTK datasets it needs
(`punkt`, `stopwords`, `wordnet`) — no manual setup required.

## Run — Console Chatbot

```bash
python chatbot.py
```

Example session:
```
You: How long does shipping take?
Bot: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.
     (matched FAQ: "How long does shipping take?" | similarity: 1.00)

You: can I get a refund?
Bot: You can return any item within 30 days of purchase for a full refund...
     (matched FAQ: "What is your return policy?" | similarity: 0.31)

You: quit
Bot: Goodbye! Have a great day.
```

## Run — Optional Web Chat UI

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Type a question in the
input box and press Enter or click Send.

## How it works

1. **Collect FAQs** — `faqs.json` holds a list of `{question, keywords, answer}`
   objects. `keywords` are extra related terms (e.g. "refund, money back" for
   the return-policy question) so that paraphrased user questions can still
   be matched correctly.
2. **Preprocess** — every FAQ question (+ its keywords) and every user query
   goes through the same pipeline:
   - lowercase everything
   - remove punctuation
   - tokenize into words (NLTK `word_tokenize`)
   - remove English stopwords (the, is, a, ...)
   - lemmatize each word to its base form (e.g. "shipping" → "ship")
3. **Vectorize & match** — all cleaned FAQ texts are converted into TF-IDF
   vectors. When a user asks a question, it's cleaned the same way, converted
   into a TF-IDF vector using the same vocabulary, and compared against every
   FAQ vector using **cosine similarity**. The FAQ with the highest similarity
   score is selected as the match.
4. **Threshold check** — if the best similarity score is below a threshold
   (default `0.25`), the bot admits it doesn't know the answer instead of
   returning a bad guess.
5. **Display answer** — the matched FAQ's answer is shown to the user, along
   with which FAQ question it matched and the similarity score (useful for
   debugging/demoing).

## Customizing for your own FAQs

Edit `faqs.json` and add your own entries in the same format:
```json
{
  "question": "Your question here?",
  "keywords": "related words synonyms alternate phrasing",
  "answer": "The answer to show the user."
}
```
No code changes needed — the chatbot rebuilds its vectorizer from
`faqs.json` every time it starts.

## Tuning match sensitivity

In `chatbot.py`, `FAQChatbot(similarity_threshold=0.25)` controls how
confident the bot must be before answering:
- **Lower** the threshold (e.g. `0.15`) → bot answers more often, but may
  give wrong matches for unrelated questions.
- **Raise** the threshold (e.g. `0.4`) → bot is more cautious and says "I
  don't know" more often, but matches it does give are more reliable.

## Possible Improvements
- Swap TF-IDF for sentence embeddings (`sentence-transformers`) for deeper
  semantic matching (e.g. correctly matching "when will my stuff arrive" to
  "shipping time" without needing manual keyword lists).
- Add spelling correction before preprocessing.
- Log unanswered questions to a file so you can expand the FAQ set over time.

## Author
Built as part of the CodeAlpha Artificial Intelligence Internship.

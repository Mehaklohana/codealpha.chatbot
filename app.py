"""
CodeAlpha - Task 2: FAQ Chatbot (Optional Web UI)
---------------------------------------------------
A minimal Flask web app providing a simple chat interface for the FAQ chatbot
defined in chatbot.py.

Usage:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot(faq_path="faqs.json")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    answer, matched_question, score = bot.get_response(user_message)
    return jsonify({
        "answer": answer,
        "matched_question": matched_question,
        "score": round(float(score), 2),
    })


if __name__ == "__main__":
    app.run(debug=True)

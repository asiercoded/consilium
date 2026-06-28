import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
from openai import OpenAI

app = Flask(__name__)
CORS(app)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data["question"]

    # AI 1 — Llama 3.3 via Groq
    response_a = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an analytical thinker. Give a clear, direct answer. Be confident in your position."},
            {"role": "user", "content": user_question}
        ]
    )
    first_ai_response = response_a.choices[0].message.content

    # AI 2 — GPT-OSS via Groq
    response_b = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a critical contrarian. You've read another AI's response and your job is to challenge it — find flaws, missing angles, or outright disagree where warranted. Don't just add on, push back."},
            {"role": "user", "content": f"Question: {user_question}\n\nAnother AI said:\n{first_ai_response}\n\nChallenge this response."}
        ]
    )
    second_ai_response = response_b.choices[0].message.content

    # AI 3 — Gemini via OpenRouter
    response_c = or_client.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=[
            {"role": "system", "content": "You are an independent thinker. Two AIs have debated a question. Read both sides and form your own distinct position — you can agree with one, disagree with both, or offer a third perspective. Be direct about where you stand."},
            {"role": "user", "content": f"Question: {user_question}\n\nAI 1 said:\n{first_ai_response}\n\nAI 2 challenged with:\n{second_ai_response}\n\nWhat's your independent take?"}
        ]
    )
    third_ai_response = response_c.choices[0].message.content

    # AI 4 — DeepSeek R1 via OpenRouter (synthesis)
    response_d = or_client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": "You are a synthesis engine. Three AIs have debated a question. Read the full debate carefully, identify the strongest arguments, resolve contradictions, and produce a comprehensive, well-structured final answer. Be thorough — don't just summarize, give the definitive answer a smart person would want to read."},
            {"role": "user", "content": f"Question: {user_question}\n\nAI 1 (Llama):\n{first_ai_response}\n\nAI 2 (GPT-OSS) challenged:\n{second_ai_response}\n\nAI 3 (Gemini) independent take:\n{third_ai_response}\n\nSynthesize into the best possible answer."}
        ]
    )
    fourth_ai_response = response_d.choices[0].message.content

    return jsonify({
        "ai1": first_ai_response,
        "ai2": second_ai_response,
        "ai3": third_ai_response,
        "synthesis": fourth_ai_response
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
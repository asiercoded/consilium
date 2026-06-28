import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELS = {
    "ai1": "llama-3.3-70b-versatile",
    "ai2": "meta-llama/llama-4-scout-17b-16e-instruct",
    "ai3": "qwen/qwen3-32b",
    "synthesis": "openai/gpt-oss-20b",
}

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data["question"]

    # AI 1 — Llama 3.3: Analytical, first take
    response_a = client.chat.completions.create(
        model=MODELS["ai1"],
        messages=[
            {"role": "system", "content": "You are an analytical thinker. Give a clear, direct answer. Be confident in your position."},
            {"role": "user", "content": user_question}
        ]
    )
    first_ai_response = response_a.choices[0].message.content

    # AI 2 — Mixtral: Challenges AI1
    response_b = client.chat.completions.create(
        model=MODELS["ai2"],
        messages=[
            {"role": "system", "content": "You are a critical contrarian. You've read another AI's response and your job is to challenge it — find flaws, missing angles, or outright disagree where warranted. Don't just add on, push back."},
            {"role": "user", "content": f"Question: {user_question}\n\nAnother AI said:\n{first_ai_response}\n\nChallenge this response."}
        ]
    )
    second_ai_response = response_b.choices[0].message.content

    # AI 3 — Gemma: Independent perspective, can side with either or take a third position
    response_c = client.chat.completions.create(
        model=MODELS["ai3"],
        messages=[
            {"role": "system", "content": "You are an independent thinker. Two AIs have debated a question. Read both sides and form your own distinct position — you can agree with one, disagree with both, or offer a third perspective. Be direct about where you stand."},
            {"role": "user", "content": f"Question: {user_question}\n\nAI 1 said:\n{first_ai_response}\n\nAI 2 challenged with:\n{second_ai_response}\n\nWhat's your independent take?"}
        ]
    )
    third_ai_response = response_c.choices[0].message.content

    # AI 4 — DeepSeek: Synthesizes the debate into a final answer
    response_d = client.chat.completions.create(
        model=MODELS["synthesis"],
        messages=[
            {"role": "system", "content": "You are a synthesis engine. Three AIs have debated a question. Your job: read the full debate, identify which arguments are strongest, resolve contradictions, and produce one clear, definitive answer. Be concise. Don't list all views — give the best answer."},
            {"role": "user", "content": f"Question: {user_question}\n\nAI 1 (Llama):\n{first_ai_response}\n\nAI 2 (Mixtral) challenged:\n{second_ai_response}\n\nAI 3 (Gemma) independent take:\n{third_ai_response}\n\nSynthesize into the best possible answer."}
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
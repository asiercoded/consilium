import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data["question"]

    response_a = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": user_question}]
    )
    first_ai_response = response_a.choices[0].message.content

    response_b = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": first_ai_response},
            {"role": "user", "content": "Build on the previous response. What important points are missing or could be explored further?"}
        ]
    )
    second_ai_response = response_b.choices[0].message.content

    response_c = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": first_ai_response},
            {"role": "user", "content": "Do you agree? What would you add?"},
            {"role": "assistant", "content": second_ai_response},
            {"role": "user", "content": "Based on the two responses above about the user's question, what unique insight or perspective would you add that hasn't been covered yet?"}
        ]
    )
    third_ai_response = response_c.choices[0].message.content

    response_d = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a synthesis AI. Read the conversation and produce one clear, concise final answer."},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": first_ai_response},
            {"role": "user", "content": "Do you agree? What would you add?"},
            {"role": "assistant", "content": second_ai_response},
            {"role": "user", "content": "Having read both responses, what would you add?"},
            {"role": "assistant", "content": third_ai_response},
            {"role": "user", "content": "Having read all three responses, synthesize everything into one clear final answer."}
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
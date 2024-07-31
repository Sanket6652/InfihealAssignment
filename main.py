from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import pipeline
import re
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("text_classifier.pkl")


class Prompt(BaseModel):
    user_prompt: str



# Sample articles
articles = [
    {
        "title": "Managing Anxiety",
        "content": "Here are some tips to manage anxiety. Practice deep breathing, exercise regularly, and seek support from friends and family.",
    },
    {
        "title": "Dealing with Depression",
        "content": "Depression can be tough, but maintaining a routine, seeking therapy, and connecting with loved ones can help.",
    },
    {
        "title": "Coping with Stress",
        "content": "Stress can be overwhelming, but practicing mindfulness, organizing your tasks, and taking breaks can be beneficial.",
    },
]


def retrieve_articles(user_prompt: str) -> List[dict]:
    # Simple keyword-based retrieval
    keywords = re.findall(r"\w+", user_prompt.lower())
    retrieved_articles = [
        article
        for article in articles
        if any(keyword in article["content"].lower() for keyword in keywords)
    ]
    return retrieved_articles


# Load the language model
generator = pipeline("text-generation", model="gpt2")


def generate_response(prompt: str, articles: List[dict]) -> str:
    # Concatenate the articles' content
    context = " ".join(article["content"] for article in articles)
    # Combine the prompt with the context
    combined_input = f"{prompt} Context: {context}"
    # Generate a response
    response = generator(combined_input, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Chatbot"}


@app.post("/rag")
def rag(prompt: Prompt):
    # Retrieve relevant articles
    retrieved_articles = retrieve_articles(prompt.user_prompt)
    if not retrieved_articles:
        raise HTTPException(status_code=404, detail="No relevant articles found")

    # Generate a response
    response = generate_response(prompt.user_prompt, retrieved_articles)
    return {"response": response, "articles": retrieved_articles}


@app.post("/classification")
def classify(prompt: Prompt):
    # Predict the label
    prediction = model.predict([prompt.user_prompt])
    return {"text": prompt.user_prompt, "label": prediction[0]}

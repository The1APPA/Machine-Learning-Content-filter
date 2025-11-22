# server_local_ml.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from better_profanity import profanity

# Initialize profanity filter
profanity.load_censor_words()  # loads default word list

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and vectorizers
bot_model = joblib.load("bot_model.pkl")
bot_vectorizer = joblib.load("bot_vectorizer.pkl")
safe_model = joblib.load("safe_model.pkl")
safe_vectorizer = joblib.load("safe_vectorizer.pkl")

class Review(BaseModel):
    name: str | None = None
    rating: int
    text: str

@app.post("/analyze")
def analyze_review(review: Review):
    text = review.text.strip()
    if not text:
        return {"appropriate": False, "message": "Empty review."}

    # 0. Check for profanity using better_profanity
    if profanity.contains_profanity(text):
        return {"appropriate": False, "message": "Profanity or offensive language detected."}

    # 1. Check for safe content via ML
    safe_vec = safe_vectorizer.transform([text])
    safe_pred = int(safe_model.predict(safe_vec)[0])

    # 2. Check for human/bot via ML
    bot_vec = bot_vectorizer.transform([text])
    bot_pred = int(bot_model.predict(bot_vec)[0])

    # Combine decisions
    if safe_pred == 0:
        return {"appropriate": False, "message": "Inappropriate or unsafe content detected."}
    if bot_pred == 0:
        return {"appropriate": False, "message": "Possible bot-generated review detected."}

    # Return approved review
    return {
        "appropriate": True,
        "review": {
            "name": review.name or "Anonymous",
            "rating": review.rating,
            "text": review.text
        }
    }

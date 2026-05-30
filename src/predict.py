import joblib
import numpy as np
import os

# -----------------------------
# Load artifacts (models + vectorizer)
# -----------------------------
class FakeNewsDetector:
    def __init__(self, model_type="svm"):
        self.model_type = model_type

        # Load vectorizer
        self.vectorizer = joblib.load("models/vectorizer.pkl")

        # Load models
        self.svm_model = joblib.load("models/svm.pkl")
        self.lr_model = joblib.load("models/logistic_regression.pkl")

    # -----------------------------
    # Preprocess + Predict
    # -----------------------------
    def predict(self, text):
        if not text or not text.strip():
            return {
                "label": "Invalid Input",
                "confidence": 0.0
            }

        # Convert text → vector
        X = self.vectorizer.transform([text])

        # Choose model
        model = self.svm_model if self.model_type == "svm" else self.lr_model

        # Prediction
        pred = model.predict(X)[0]

        # Confidence score (if available)
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(X)))
        else:
            confidence = None

        label = "Fake News" if pred == 1 else "Real News"

        return {
            "label": label,
            "confidence": confidence
        }


# -----------------------------
# Example usage (CLI test)
# -----------------------------
if __name__ == "__main__":
    detector = FakeNewsDetector(model_type="svm")

    print("\n📰 Fake News Detection System")
    print("-" * 40)

    while True:
        text = input("\nEnter news text (or type 'exit'): ")

        if text.lower() == "exit":
            break

        result = detector.predict(text)

        print("\n🔍 Prediction:", result["label"])

        if result["confidence"] is not None:
            print(f"📊 Confidence: {result['confidence']:.4f}")

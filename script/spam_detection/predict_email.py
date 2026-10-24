
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn


# ============================================
# ECS 170 - Spam Email Detection
# Student: Yourui Yang
# Student ID: 925235594
# ============================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "enron_spam_data.csv"
MODEL_PATH = ROOT / "results" / "best_spam_mlp.pt"

SEED = 42
DEVICE = torch.device("cpu")


# ============================================
# 1. Neural Network Architecture
# ============================================

class SpamMLP(nn.Module):

    def __init__(
        self,
        input_dim=5000,
        h1=128,
        h2=64,
        dropout=0.0
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================
# 2. Rebuild Original TF-IDF Vectorizer
# ============================================

def load_vectorizer():

    print("Loading email dataset...")

    df = pd.read_csv(DATA_PATH)

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    df["text"] = (
        df["Subject"] + " " + df["Message"]
    ).str.strip()

    labels = (
        df["Spam/Ham"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["label"] = labels.map({
        "spam": 1,
        "ham": 0
    })

    df = df.dropna(subset=["label"])

    df = df[df["text"] != ""]

    df = df.drop_duplicates(subset=["text"])

    # Same split used in run_experiments.py
    X_train, X_test, y_train, y_test = (
        train_test_split(
            df["text"],
            df["label"].astype(int),
            test_size=0.2,
            random_state=SEED,
            stratify=df["label"]
        )
    )

    X_fit, X_val, y_fit, y_val = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=SEED,
            stratify=y_train
        )
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    # Fit only on the original training subset
    vectorizer.fit(X_fit)

    print("TF-IDF vectorizer ready!")

    return vectorizer


# ============================================
# 3. Load Trained MLP Model
# ============================================

def load_model(input_dim):

    print("Loading trained MLP model...")

    model = SpamMLP(
        input_dim=input_dim,
        h1=128,
        h2=64,
        dropout=0.0
    ).to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)

    model.eval()

    print("MLP model loaded successfully!")

    return model


# ============================================
# 4. Predict Spam / Ham
# ============================================

def predict_email(text, model, vectorizer):

    features = vectorizer.transform([text])

    features = features.toarray().astype(
        np.float32
    )

    tensor = torch.from_numpy(features).to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(tensor)

        probability = torch.sigmoid(
            logits
        ).item()

    if probability >= 0.5:
        prediction = "SPAM"
    else:
        prediction = "HAM"

    return prediction, probability


# ============================================
# 5. Interactive Email Detector
# ============================================

def main():

    print("=" * 55)
    print("ECS 170 - AI Spam Email Detection System")
    print("Student: Yourui Yang")
    print("=" * 55)

    if not DATA_PATH.exists():
        print("ERROR: Dataset not found!")
        print(DATA_PATH)
        sys.exit(1)

    if not MODEL_PATH.exists():
        print("ERROR: Trained model not found!")
        print(MODEL_PATH)
        sys.exit(1)

    vectorizer = load_vectorizer()

    input_dim = len(
        vectorizer.get_feature_names_out()
    )

    model = load_model(input_dim)

    print("\n" + "=" * 55)
    print("SYSTEM READY")
    print("=" * 55)

    print("Enter an English email to classify.")
    print("Type 'exit' to quit.\n")

    while True:

        print("-" * 55)

        email = input(
            "Enter email text: "
        ).strip()

        if email.lower() == "exit":
            print("\nGoodbye!")
            break

        if not email:
            print("Please enter an email.")
            continue

        prediction, probability = predict_email(
            email,
            model,
            vectorizer
        )

        print("\n========== PREDICTION ==========")

        print("Classification:", prediction)

        print(
            f"Spam Probability: {probability:.2%}"
        )

        print(
            f"Ham Probability: {1 - probability:.2%}"
        )

        if prediction == "SPAM":
            print(
                "\nWarning: Potential spam email detected!"
            )
        else:
            print(
                "\nThis email is classified as legitimate."
            )

        print("================================\n")


# ============================================
# Run Program
# ============================================

if __name__ == "__main__":
    main()

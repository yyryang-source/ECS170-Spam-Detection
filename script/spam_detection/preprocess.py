
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "enron_spam_data.csv"


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns.tolist())

    # Combine subject and message
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    df["text"] = (
        df["Subject"] + " " + df["Message"]
    ).str.strip()

    # Convert labels to binary values
    labels = df["Spam/Ham"].astype(str).str.lower().str.strip()
    df["label"] = labels.map({"spam": 1, "ham": 0})

    # Remove invalid and duplicate records
    df = df.dropna(subset=["label"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])

    X = df["text"]
    y = df["label"].astype(int)

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Learn vocabulary only from training data
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTotal emails:", len(df))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training feature shape:", X_train_tfidf.shape)
    print("Testing feature shape:", X_test_tfidf.shape)

    return (
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test
    )


if __name__ == "__main__":
    prepare_data()

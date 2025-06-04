'''
This file determines the urgency of each query based on a heuristic rule.
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Heuristic Labeling Function
def label_priority(query):
    query = query.lower()

    if any(word in query for word in ["urgent", "respond quickly", "ASAP", "time-sensitive", "important, "time-critical"]):
        return "high"
    elif any(word in query for word in ["can you tell me", "could you", "please tell me", "i'd like to know", "what kind of", "what is in", "what does this", "do you see"]):
        return "medium"
    else:
        return "low"

# Load and Label Dataset 
df = pd.read_csv("/content/questions.csv")  # Please revise to your path
df = df.dropna(subset=["question"])
df["priority"] = df["question"].apply(label_priority)

# Encode priority labels
priority_map = {"low": 0, "medium": 1, "high": 2}
df["priority_label"] = df["priority"].map(priority_map)

# Model Training
X_train, X_test, y_train, y_test = train_test_split(
    df["question"], df["priority_label"], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
inv_map = {v: k for k, v in priority_map.items()}
target_names = [inv_map[i] for i in sorted(inv_map)]
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Predict on Full Dataset
df["predicted_priority_label"] = model.predict(df["question"])
df["predicted_priority"] = df["predicted_priority_label"].map(inv_map)

# Save Full Prediction Output
output_path = "/content/questions_with_predictions.csv"  # Please revise to your path
df.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")

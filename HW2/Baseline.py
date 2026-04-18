import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

#pip install torch transformers datasets scikit-learn pandas accelerate bitsandbytes

# 1. Load Data (Assuming a CSV with 'text' and 'label' columns)
# Students should download the dataset from Kaggle
df = pd.read_csv("train_v2_drcat_02.csv")  # Example filename
df = df[['text', 'label']] # Ensure columns exist

# Split Data
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 2. TF-IDF + Logistic Regression Baseline
print("Training TF-IDF Baseline...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train_tfidf, y_train)

# Evaluation
preds = clf.predict_proba(X_val_tfidf)[:, 1]
auc = roc_auc_score(y_val, preds)
print(f"Baseline TF-IDF ROC-AUC: {auc:.4f}")
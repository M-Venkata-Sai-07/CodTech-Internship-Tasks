# ============================================================
# TASK 4 - SENTIMENT ANALYSIS USING NLP
# CodTech Internship - Mallavarapu Venkata Sai (CTIS1591)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              accuracy_score,
                              ConfusionMatrixDisplay,
                              precision_score,
                              recall_score,
                              f1_score)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

# ── Paths ────────────────────────────────────────────────────
DATA_PATH   = "../data/training.1600000.processed.noemoticon.csv"
OUTPUTS_DIR = "../outputs/"
MODELS_DIR  = "../models/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print("=" * 60)
print("   TASK 4 - SENTIMENT ANALYSIS (NLP)")
print("=" * 60)

# ── 1. Load Data ─────────────────────────────────────────────
print("\n📂 Loading Sentiment140 dataset...")

cols = ["sentiment","id","date","query","user","text"]

# ✅ FIX: Read 400k rows (200k pos + 200k neg) to get both classes
# The file is sorted: first 800k = Negative, last 800k = Positive
df_neg = pd.read_csv(DATA_PATH, encoding="latin1",
                     names=cols, nrows=200000)
df_pos = pd.read_csv(DATA_PATH, encoding="latin1",
                     names=cols, skiprows=1400000, nrows=200000)

df = pd.concat([df_neg, df_pos], ignore_index=True)
df["sentiment"] = df["sentiment"].map({0:"Negative", 4:"Positive"})

# Drop any unmapped rows
df = df[df["sentiment"].isin(["Positive","Negative"])]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Shape     : {df.shape}")
print(f"   Positive  : {(df['sentiment']=='Positive').sum():,}")
print(f"   Negative  : {(df['sentiment']=='Negative').sum():,}")

# ── 2. EDA Plot 1: Distribution ──────────────────────────────
print("\n📊 Generating EDA plots...")
colors = ["#e74c3c", "#2ecc71"]
counts = df["sentiment"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sentiment Distribution", fontsize=14, fontweight='bold')

axes[0].bar(counts.index, counts.values, color=colors, alpha=0.85)
axes[0].set_title("Count of Tweets")
axes[0].set_ylabel("Count")
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f"{v:,}", ha='center', fontweight='bold')

# ✅ FIX: explode length matches number of classes
n_classes = len(counts)
explode   = [0.05] * n_classes
axes[1].pie(counts.values, labels=counts.index,
            autopct="%1.1f%%", colors=colors[:n_classes],
            startangle=90, explode=explode)
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}01_sentiment_distribution.png", dpi=150)
plt.close()
print("   ✅ 01_sentiment_distribution.png")

# ── 3. Tweet Length Analysis ─────────────────────────────────
df["tweet_length"] = df["text"].apply(len)
df["word_count"]   = df["text"].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Tweet Length Analysis", fontsize=14, fontweight='bold')

for sentiment, color in zip(["Positive","Negative"], ["#2ecc71","#e74c3c"]):
    subset = df[df["sentiment"] == sentiment]
    axes[0].hist(subset["tweet_length"], bins=40,
                 alpha=0.6, label=sentiment, color=color)
    axes[1].hist(subset["word_count"], bins=30,
                 alpha=0.6, label=sentiment, color=color)

axes[0].set_title("Character Length Distribution")
axes[0].set_xlabel("Characters")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].set_title("Word Count Distribution")
axes[1].set_xlabel("Words")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}02_tweet_length_analysis.png", dpi=150)
plt.close()
print("   ✅ 02_tweet_length_analysis.png")

# ── 4. Text Preprocessing ────────────────────────────────────
print("\n⚙️  Preprocessing tweets...")
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(w) for w in text.split()
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df["clean_text"] = df["text"].apply(clean_text)
print("   ✅ Text cleaning complete")

# ── 5. WordClouds ────────────────────────────────────────────
print("   Generating WordClouds...")
pos_text = ' '.join(df[df["sentiment"]=="Positive"]["clean_text"])
neg_text = ' '.join(df[df["sentiment"]=="Negative"]["clean_text"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Word Clouds by Sentiment", fontsize=14, fontweight='bold')

wc_pos = WordCloud(width=800, height=400, background_color="white",
                   colormap="Greens", max_words=100).generate(pos_text)
axes[0].imshow(wc_pos, interpolation="bilinear")
axes[0].set_title("Positive Tweets", fontsize=13)
axes[0].axis("off")

wc_neg = WordCloud(width=800, height=400, background_color="white",
                   colormap="Reds", max_words=100).generate(neg_text)
axes[1].imshow(wc_neg, interpolation="bilinear")
axes[1].set_title("Negative Tweets", fontsize=13)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}03_wordclouds.png", dpi=150)
plt.close()
print("   ✅ 03_wordclouds.png")

# ── 6. Top Words ─────────────────────────────────────────────
from collections import Counter

def top_words(text, n=15):
    words = text.split()
    return pd.DataFrame(Counter(words).most_common(n),
                        columns=["word","count"])

pos_top = top_words(pos_text)
neg_top = top_words(neg_text)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Top 15 Words by Sentiment", fontsize=14, fontweight='bold')

axes[0].barh(pos_top["word"], pos_top["count"], color="#2ecc71", alpha=0.85)
axes[0].set_title("Positive Tweets")
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(neg_top["word"], neg_top["count"], color="#e74c3c", alpha=0.85)
axes[1].set_title("Negative Tweets")
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}04_top_words.png", dpi=150)
plt.close()
print("   ✅ 04_top_words.png")

# ── 7. TF-IDF Vectorization ──────────────────────────────────
print("\n🔢 Vectorizing with TF-IDF...")
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"   Train : {X_train_tfidf.shape}")
print(f"   Test  : {X_test_tfidf.shape}")

# ── 8. Train Models ──────────────────────────────────────────
print("\n🤖 Training models...")
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, n_jobs=-1),
    "Naive Bayes"         : MultinomialNB(),
    "Linear SVM"          : LinearSVC(max_iter=2000),
}

results = {}
trained = {}
for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train_tfidf, y_train)
    preds          = model.predict(X_test_tfidf)
    acc            = accuracy_score(y_test, preds)
    results[name]  = acc
    trained[name]  = (model, preds)
    print(f"   ✅ {name}: Accuracy = {acc:.4f}")

# ── 9. Model Accuracy Comparison ────────────────────────────
plt.figure(figsize=(9, 5))
bars = plt.bar(results.keys(), results.values(),
               color=["#3498db","#f39c12","#2ecc71"], alpha=0.85)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy")
plt.ylim(0.70, 1.00)
for bar, acc in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.003,
             f"{acc:.4f}", ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}05_model_accuracy.png", dpi=150)
plt.close()
print("\n   ✅ 05_model_accuracy.png")

# ── 10. Best Model ───────────────────────────────────────────
best_name           = max(results, key=results.get)
best_model, best_preds = trained[best_name]
print(f"\n🏆 Best Model: {best_name} (Accuracy={results[best_name]:.4f})")
print("\n   Classification Report:")
print(classification_report(y_test, best_preds))

# ── 11. Confusion Matrix ─────────────────────────────────────
cm   = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative","Positive"])
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}06_confusion_matrix.png", dpi=150)
plt.close()
print("   ✅ 06_confusion_matrix.png")

# ── 12. Full Metrics Comparison ──────────────────────────────
metrics_data = {}
for name, (model, preds) in trained.items():
    metrics_data[name] = {
        "Accuracy"  : accuracy_score(y_test, preds),
        "Precision" : precision_score(y_test, preds, pos_label="Positive"),
        "Recall"    : recall_score(y_test, preds,    pos_label="Positive"),
        "F1 Score"  : f1_score(y_test, preds,        pos_label="Positive"),
    }

metrics_df = pd.DataFrame(metrics_data).T
x     = np.arange(len(metrics_df))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, (metric, color) in enumerate(zip(
        ["Accuracy","Precision","Recall","F1 Score"],
        ["#3498db","#e74c3c","#2ecc71","#f39c12"])):
    ax.bar(x + i*width, metrics_df[metric],
           width, label=metric, color=color, alpha=0.85)

ax.set_title("Full Metrics Comparison", fontsize=14, fontweight='bold')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_df.index, rotation=10)
ax.set_ylabel("Score")
ax.set_ylim(0.70, 1.00)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}07_full_metrics.png", dpi=150)
plt.close()
print("   ✅ 07_full_metrics.png")

# ── 13. Live Predictions ─────────────────────────────────────
print("\n🔮 Live Predictions:")
test_tweets = [
    "I love this! Absolutely amazing experience!",
    "This is the worst thing ever. So disappointing.",
    "Just had a great day with friends!",
    "Terrible service, never going back there.",
    "The movie was fantastic, highly recommend!",
    "So frustrated, nothing works properly.",
    "Beautiful weather today, feeling blessed!",
    "Lost my keys again, worst day ever.",
]

print(f"\n   {'Tweet':<47} Sentiment")
print("   " + "-" * 60)
for tweet in test_tweets:
    cleaned    = clean_text(tweet)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    emoji      = "😊" if prediction == "Positive" else "😞"
    print(f"   {tweet[:46]:<47} {emoji} {prediction}")

# ── 14. Save Models ──────────────────────────────────────────
print("\n💾 Saving models and reports...")
joblib.dump(best_model, f"{MODELS_DIR}sentiment_model.pkl")
joblib.dump(tfidf,      f"{MODELS_DIR}tfidf_vectorizer.pkl")
metrics_df.to_csv(f"{OUTPUTS_DIR}results.csv")
print("   ✅ sentiment_model.pkl")
print("   ✅ tfidf_vectorizer.pkl")
print("   ✅ results.csv")

# ── 15. Summary ──────────────────────────────────────────────
outputs = os.listdir(OUTPUTS_DIR)
print("\n" + "=" * 60)
print("   TASK 4 COMPLETE!")
print("=" * 60)
print(f"\n📁 Output files ({len(outputs)}):")
for f in sorted(outputs):
    size = os.path.getsize(f"{OUTPUTS_DIR}{f}")
    print(f"   {f} ({size/1024:.1f} KB)")
print(f"\n🏆 Best Model : {best_name}")
print(f"   Accuracy   : {results[best_name]:.4f}")
print("\n✅ Sentiment Analysis Complete!")
print("=" * 60)
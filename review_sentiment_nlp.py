import re
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CREATE SYNTHETIC PRODUCT REVIEW DATASET
# ============================================================

positive_reviews = [
    "This product works perfectly and I am very satisfied.",
    "Excellent quality and very good performance.",
    "Amazing product and definitely worth the money.",
    "I really love the design and build quality.",
    "Fantastic purchase and very easy to use.",
    "The product exceeded my expectations.",
    "Great performance and excellent value for money.",
    "Very happy with this product and would buy again.",
    "The quality is impressive and delivery was fast.",
    "Highly recommended product with excellent features.",
    "Works exactly as described and feels premium.",
    "Very reliable and performs extremely well.",
    "I had a wonderful experience using this product.",
    "Great build quality and smooth operation.",
    "This is one of the best products I have purchased.",
    "Excellent customer experience and good quality.",
    "The device performs well and is easy to operate.",
    "Very useful product with impressive performance.",
    "I am completely satisfied with my purchase.",
    "The product arrived quickly and works beautifully.",
    "Good value and reliable performance.",
    "The product feels durable and well designed.",
    "Really impressed with the overall quality.",
    "Everything works properly and setup was simple.",
    "A great choice for everyday use.",
    "Performance has been consistently good.",
    "The item matches the description perfectly.",
    "I would happily recommend this product to others.",
    "Very convenient and easy to understand.",
    "The product has been working flawlessly.",
    "Good packaging and excellent product quality.",
    "I am pleased with how well this performs.",
    "This purchase was absolutely worth it.",
    "The product is efficient and dependable.",
    "Great functionality for the price.",
    "Using this product has been a positive experience.",
    "The quality is better than I expected.",
    "Everything about the product works nicely.",
    "Really satisfied with the features provided.",
    "A dependable product that does its job well."
]

negative_reviews = [
    "This product is terrible and stopped working quickly.",
    "Very poor quality and a complete waste of money.",
    "I am extremely disappointed with this purchase.",
    "The product arrived damaged and does not work.",
    "Terrible performance and very unreliable.",
    "I regret buying this product.",
    "The quality is awful and feels very cheap.",
    "Worst product I have purchased in a long time.",
    "The device crashes constantly and is frustrating.",
    "Poor build quality and disappointing performance.",
    "The product failed after only a few days.",
    "Customer support was terrible and unhelpful.",
    "Not worth the price and very poorly designed.",
    "The item does not work as advertised.",
    "Very disappointing experience overall.",
    "Bad quality and unreliable operation.",
    "The device freezes frequently during normal use.",
    "I would never recommend this product.",
    "The product has too many problems.",
    "Completely dissatisfied with the purchase.",
    "Setup was difficult and the product barely works.",
    "The item feels fragile and badly constructed.",
    "Performance is inconsistent and frustrating.",
    "The product was not worth what I paid.",
    "It stopped functioning shortly after installation.",
    "Very disappointing quality for the price.",
    "I experienced several problems while using it.",
    "The product does not perform properly.",
    "The overall experience has been very poor.",
    "I expected much better quality than this.",
    "The item arrived late and was also defective.",
    "Nothing about this product impressed me.",
    "The performance became worse after a few uses.",
    "It is difficult to use and poorly designed.",
    "The product feels unreliable during daily use.",
    "I am unhappy with the quality I received.",
    "Several features do not work correctly.",
    "This purchase turned out to be disappointing.",
    "The device is slow and frequently stops responding.",
    "I would choose a different product next time."
]

reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

df = pd.DataFrame({
    "Review": reviews,
    "Sentiment": labels
})


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove unnecessary spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Cleaned_Review"] = df["Review"].apply(clean_text)


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X = df["Cleaned_Review"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    C=1.0,
    solver="liblinear",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# 6. TEST SET EVALUATION
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("E-COMMERCE REVIEW SENTIMENT ANALYZER")
print("=" * 60)

print(f"\nTotal Reviews: {len(df)}")
print(f"Training Reviews: {len(X_train)}")
print(f"Testing Reviews: {len(X_test)}")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)


# ============================================================
# 7. 5-FOLD CROSS-VALIDATION
# ============================================================

# Pipeline prevents TF-IDF from being fitted on validation data
cv_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            random_state=42
        )
    )
])

cv_scores = cross_val_score(
    cv_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation Scores:")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold}: {score * 100:.2f}%")

print(
    f"Average Cross-Validation Accuracy: "
    f"{cv_scores.mean() * 100:.2f}%"
)


# ============================================================
# 8. TEST COMPLETELY NEW REVIEWS
# ============================================================

new_reviews = [
    "The product works really well and I am happy with it.",
    "The device stopped working and I am disappointed.",
    "Delivery was slow but the product itself works well.",
    "The design looks good but performance is disappointing."
]

print("\nNew Review Predictions:")

for review in new_reviews:

    cleaned_review = clean_text(review)

    review_tfidf = vectorizer.transform([cleaned_review])

    prediction = model.predict(review_tfidf)[0]

    probability = model.predict_proba(review_tfidf)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    confidence = probability[prediction] * 100

    print("\nReview:", review)
    print("Predicted Sentiment:", sentiment)
    print(f"Model Confidence: {confidence:.2f}%")


print("\n" + "=" * 60)

# ============================================================
# 9. BASELINE MODEL COMPARISON
# ============================================================

from sklearn.naive_bayes import MultinomialNB

nb_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

nb_scores = cross_val_score(
    nb_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nNaive Bayes 5-Fold Cross-Validation:")

for fold, score in enumerate(nb_scores, start=1):
    print(f"Fold {fold}: {score * 100:.2f}%")

print(
    f"Naive Bayes Average Accuracy: "
    f"{nb_scores.mean() * 100:.2f}%"
)

print("\nModel Comparison:")
print(
    f"Logistic Regression CV Accuracy: "
    f"{cv_scores.mean() * 100:.2f}%"
)

print(
    f"Naive Bayes CV Accuracy: "
    f"{nb_scores.mean() * 100:.2f}%"
)
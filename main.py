import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- LOAD & CLEAN DATA ----------
print("Loading dataset... ⏳")

data = pd.read_csv("fake_real.csv")

# Clean data
data = data.dropna(subset=["text", "target"])

# Features & Labels
X = data["text"]
y = data["target"]

# Convert text to numbers
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully ✅")

# ---------- FUNCTIONS ----------

def check_news():
    news = input("\nEnter news text:\n").strip()

    if news == "":
        print("⚠️ Empty input! Try again.")
        return

    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)
    prob = model.predict_proba(news_vec)

    confidence = max(prob[0]) * 100

    if result[0] == 0:
        print(f"\n🟥 Fake News ({confidence:.2f}% confidence)")
    else:
        print(f"\n🟩 Real News ({confidence:.2f}% confidence)")


def show_help():
    print("\n===== HELP =====")
    print("1. Choose 'Check News' to input any news text.")
    print("2. The system will predict whether it is Fake or Real.")
    print("3. Confidence score is also shown.")
    print("4. Choose Exit to close the program.")


# ---------- CLI MENU ----------
while True:
    print("\n===== SMART FAKE NEWS DETECTOR =====")
    print("1. Check News")
    print("2. Help")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        check_news()

    elif choice == '2':
        show_help()

    elif choice == '3':
        print("Exiting program... 👋")
        break

    else:
        print("❌ Invalid choice! Please enter 1, 2 or 3.")
import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

print(df.shape)        # how many rows & columns
print(df.head())       # first 5 rows
print(df['sentiment'].value_counts())  # positive vs negative count

df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(df['label'].value_counts())

import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)      # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
    text = text.lower()                       # lowercase everything
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)
print(df['clean_review'][0])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['label']

print(X.shape)  # should be (50000, 10000)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training complete!")

sample = ["This movie was brilliant. I loved every moment."]
sample_clean = [clean_text(sample[0])]
sample_vec = vectorizer.transform(sample_clean)
pred = model.predict(sample_vec)
print("Positive" if pred[0] == 1 else "Negative")

from sklearn.metrics import (accuracy_score,
    classification_report, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
      target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model saved!")
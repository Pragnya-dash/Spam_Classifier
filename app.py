import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict(text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    return "🚫 Spam" if result[0] == 1 else "✅ Not Spam"

# UI
st.title("📧 Email Spam Classifier")
st.write("Enter your message below:")

user_input = st.text_area("Message")

if st.button("Check"):
    if user_input.strip() != "":
        result = predict(user_input)
        st.success(result)
    else:
        st.warning("Please enter a message")
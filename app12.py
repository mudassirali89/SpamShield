import streamlit as st
import string
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from telegram import Update
nltk.download('punkt_tab')

import multiprocessing
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Download necessary NLTK data
nltk.download('punkt')
ps = PorterStemmer()

# Load pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Classification function
def classify_sms(text):
    transformed_sms = transform_text(text)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return result

# Telegram bot token
BOT_TOKEN = 'enter_bot_token_here'

# Telegram bot start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Your SMS Spam Classifier bot is ready.")

# Telegram bot message handler
async def classify_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_text = update.message.text
    result = classify_sms(message_text)

    if result == 1:
        await update.message.reply_text("Warning: This SMS appears to be spam.")
    else:
        await update.message.reply_text("This SMS is not spam.")

# Main function for the Telegram bot
def main():
    # Initialize the application with the bot token
    app = Application.builder().token(BOT_TOKEN).build()

    # Register command and message handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify_message))

    # Run the bot
    app.run_polling()

# Entry point
if __name__ == '__main__':
    # Run the Telegram bot in a separate process
    multiprocessing.Process(target=main).start()

    # Streamlit app code
    st.title("SMS Spam Classifier")
    input_sms = st.text_area("Enter the message")
    if st.button('Predict'):
        result = classify_sms(input_sms)

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

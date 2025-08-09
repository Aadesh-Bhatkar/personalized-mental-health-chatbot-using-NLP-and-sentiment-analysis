# app.py
# Personalized Mental Health Chatbot (Streamlit + NLTK VADER + SQLite)

import os
import re
import random
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------
# Setup
# ---------------------------
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

DB_PATH = os.path.join(os.getcwd(), 'chat_history.db')

# ---------------------------
# Configurable constants
# ---------------------------
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'harm myself',
    'want to die', "i can't go on", "i cant go on"
]

COMMON_KEYWORDS = {
    'stress': ['stress', 'stressed', 'overwhelmed'],
    'anxiety': ['anxious', 'anxiety', 'panic', 'worry'],
    'depressed': ['depress', 'sad', 'unhappy', 'hopeless', 'down'],
    'sleep': ['insomnia', 'sleep', 'tired', 'sleeping'],
    'exam': ['exam', 'test', 'interview', 'deadline']
}

POSITIVE_RESPONSES = [
    "That's wonderful to hear — keep that energy going! Anything you want to celebrate?",
    "Awesome! I'm glad things are going well. Want to share more?",
    "Great! Celebrating small wins is powerful — tell me one thing that went well today."
]

NEUTRAL_RESPONSES = [
    "I see. Tell me a bit more about what's on your mind.",
    "Okay — would you like a breathing exercise or a quick mood check?",
    "I understand. Want to try a short grounding exercise together?"
]

NEGATIVE_RESPONSES = [
    "I'm sorry you're feeling this way. Would you like a simple breathing exercise or a coping tip?",
    "That sounds tough. I'm here for you — do you want a grounding exercise or a small distraction?",
    "I'm listening. If you want, we can try a 1-minute breathing exercise together."
]

COPING_TIPS = [
    "Try the 4-4-4 breathing: inhale 4s, hold 4s, exhale 4s. Do this for 1-2 minutes.",
    "Take a short walk, even 5–10 minutes. Movement can help reset your mood.",
    "Write down 3 things you did well today — little wins matter.",
    "Listen to a calming song you like for 5 minutes or try a guided breathing app."
]

CRISIS_MESSAGE = (
    "I’m really sorry you’re feeling this way. If you are in immediate danger or think you might "
    "harm yourself, please contact your local emergency services right now. "
    "If you can, reach out to someone you trust or a mental health professional."
)

# ---------------------------
# Helper functions — NLP
# ---------------------------
def preprocess(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

def sentiment_scores(text: str) -> dict:
    return sia.polarity_scores(preprocess(text))

def contains_crisis(text: str) -> bool:
    t = preprocess(text)
    return any(re.search(rf'\b{re.escape(kw)}\b', t) for kw in CRISIS_KEYWORDS)

def extract_tags(text: str):
    t = preprocess(text)
    return [tag for tag, words in COMMON_KEYWORDS.items() if any(w in t for w in words)]

def pick_response(sentiment: str):
    if sentiment == 'positive':
        return random.choice(POSITIVE_RESPONSES)
    elif sentiment == 'neutral':
        return random.choice(NEUTRAL_RESPONSES)
    return random.choice(NEGATIVE_RESPONSES)

def get_coping_tip():
    return random.choice(COPING_TIPS)

# ---------------------------
# Helper functions — Database
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            role TEXT,
            text TEXT,
            compound REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_message(user, role, text, compound=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO messages (user, role, text, compound, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user, role, text, compound, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_history(user, limit=500):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT role, text, compound, timestamp FROM messages
        WHERE user = ?
        ORDER BY id DESC
        LIMIT ?
    ''', (user, limit)).fetchall()
    conn.close()
    return list(reversed(rows))

# ---------------------------
# Initialize
# ---------------------------
init_db()
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🫶")
st.title("🫶 Personal Mental Health Chatbot")

# ---------------------------
# Sidebar
# ---------------------------
username = st.sidebar.text_input("Your name (for this session)", value="Guest")
if not username:
    st.sidebar.warning("Enter a name to save your chat history.")

st.sidebar.markdown("*Safety note:* This bot is supportive, not a replacement for professional care.")
st.sidebar.markdown("If you are in immediate danger, contact local emergency services.")
show_mood = st.sidebar.checkbox("Show mood history chart", value=True)
show_history = st.sidebar.checkbox("Show full chat history", value=False)

# ---------------------------
# Conversation display
# ---------------------------
st.subheader("Conversation")
history = get_history(username)
if history:
    for role, text, _, _ in history:
        st.markdown(f"{'You' if role == 'user' else 'Bot'}:** {text}")
else:
    st.info("No chat history yet. Start a conversation below.")

# ---------------------------
# Input
# ---------------------------
user_input = st.text_input("Say something (type 'exit' to end):", key="input")
if st.button("Send") and user_input:
    scores = sentiment_scores(user_input)
    compound = scores['compound']
    save_message(username, 'user', user_input, compound)

    if contains_crisis(user_input) or compound <= -0.85:
        bot_text = CRISIS_MESSAGE
        save_message(username, 'bot', bot_text)
        st.error(bot_text)
        st.info("If you're in India, call AASRA 91-22-2754-6669, or your local emergency number.")
    else:
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        tags = extract_tags(user_input)
        bot_text = pick_response(sentiment)

        if sentiment == 'negative':
            bot_text += " Tip: " + get_coping_tip()
        elif sentiment == 'neutral' and tags:
            bot_text += f" I noticed you're talking about {', '.join(tags)}. Want tips related to that?"

        save_message(username, 'bot', bot_text)
        st.write(f"*Bot:* {bot_text}")

    st.rerun()

# ---------------------------
# Mood chart
# ---------------------------
if show_mood:
    rows = get_history(username, limit=1000)
    df = pd.DataFrame(
        [{'timestamp': r[3], 'compound': r[2]} for r in rows if r[0] == 'user']
    )
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mood_df = df.set_index('timestamp').resample('D').mean().fillna(method='ffill')
        st.subheader("Mood trend (daily average sentiment)")
        st.line_chart(mood_df)
    else:
        st.info("No mood history yet. Start chatting!")

# ---------------------------
# Show raw history
# ---------------------------
if show_history:
    all_rows = get_history(username, limit=1000)
    if all_rows:
        st.subheader("Raw Chat History")
        hist_df = pd.DataFrame(all_rows, columns=['role', 'text', 'compound', 'timestamp'])
        st.dataframe(hist_df)
    else:
        st.info("No history to show.")

st.markdown("---")
st.markdown("*Note:* Data is stored locally in chat_history.db. Not for clinical use.")

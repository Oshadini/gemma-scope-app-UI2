import streamlit as st
import requests
import pandas as pd
import altair as alt
import re

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/search-all"
MODEL_ID = "gpt2-small"
SOURCE_SET = "res-jb"
SELECTED_LAYERS = ["6-res-jb"]
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": "sk-np-h0ZsR5M1gY0w8al332rJUYa0C8hQL2yUogd5n4Pgvvg0"
}

# Initialize Session State
if "selected_token" not in st.session_state:
    st.session_state.selected_token = None
if "tokens" not in st.session_state:
    st.session_state.tokens = []

# Helper Functions
def tokenize_sentence(sentence):
    """Tokenize the input sentence using regex."""
    return re.findall(r"\b\w+\b|[^\w\s]", sentence)

def fetch_explanations_for_token(token):
    """Fetch explanations from Neuronpedia API for a given token."""
    payload = {
        "modelId": MODEL_ID,
        "sourceSet": SOURCE_SET,
        "text": token,
        "selectedLayers": SELECTED_LAYERS,
        "sortIndexes": [1],
        "ignoreBos": False,
        "densityThreshold": -1,
        "numResults": 50,
    }
    try:
        response = requests.post(NEURONPEDIA_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        return response.json().get("result", [])
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

# App Layout and Styling
st.set_page_config(page_title="Token Feature Analysis", layout="wide", page_icon="🔍")

# Custom CSS
st.markdown(
    """
    <style>
    .token-container {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .stButton > button {
        height: 40px;
        width: auto;
        background-color: #007acc;
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 14px;
        font-weight: bold;
        padding: 0 10px;
    }
    .stButton > button:hover {
        background-color: #005f99;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    "<h1 style='text-align:center; color:#1F618D;'>Token Feature Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)

# Input Section
st.sidebar.markdown("<h3 style='color:#1ABC9C;'>Input Sentence</h3>", unsafe_allow_html=True)
sentence = st.sidebar.text_area("Enter a sentence:")
if st.sidebar.button("Generate Tokens"):
    st.session_state.tokens = tokenize_sentence(sentence)

# Token Display
if st.session_state.tokens:
    st.markdown("<h2 style='color:#1F618D;'>Sentence Tokenization</h2>", unsafe_allow_html=True)
    st.markdown('<div class="token-container">', unsafe_allow_html=True)
    for idx, token in enumerate(st.session_state.tokens):
        if st.button(token, key=f"token_{idx}"):
            st.session_state.selected_token = token
    st.markdown('</div>', unsafe_allow_html=True)

# Token Feature Display
if st.session_state.selected_token:
    st.markdown(f"<h3 style='color:#1ABC9C;'>Features for Token: <em>{st.session_state.selected_token}</em></h3>", unsafe_allow_html=True)
    explanations = fetch_explanations_for_token(st.session_state.selected_token)

    if explanations:
        st.markdown("### Available Features:")
        for explanation in explanations[:5]:  # Display up to 5 features
            st.write(f"- {explanation.get('description', 'No description available')}")
    else:
        st.warning("No explanations found for the selected token.")

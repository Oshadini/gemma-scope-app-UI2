# File: sentence_token_features_ui_with_advanced_ui.py

import streamlit as st
import requests
import pandas as pd
import altair as alt
import re  # Import regex for improved tokenization

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/search-all"
MODEL_ID = "gpt2-small"
SOURCE_SET = "res-jb"
SELECTED_LAYERS = ["6-res-jb"]
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": "YOUR_TOKEN"  # Replace with your actual API token
}

# Initialize Session State
if "selected_token" not in st.session_state:
    st.session_state.selected_token = None
if "available_explanations" not in st.session_state:
    st.session_state.available_explanations = []
if "selected_explanation" not in st.session_state:
    st.session_state.selected_explanation = {}

# Helper Functions
def tokenize_sentence(sentence):
    """Tokenize the input sentence using regex for better handling of punctuation and whitespace."""
    return re.findall(r"\b\w+\b|[^\w\s]", sentence)

def fetch_features_for_token(token):
    """Fetch features for a given token using the 'Top Features for Text' API."""
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
        result = response.json()
        explanations = result.get("results", [])
        return explanations
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

def plot_graph(x_data, y_data, title, x_label="X-axis", y_label="Y-axis"):
    """Generate a histogram for visualization."""
    if not x_data or not y_data:
        st.write(f"No data available for {title}.")
        return None
    chart = alt.Chart(pd.DataFrame({"x": x_data, "y": y_data})).mark_bar(color="#A3E4D7").encode(
        x=alt.X("x:Q", title=x_label, axis=alt.Axis(labelColor="#117864")),
        y=alt.Y("y:Q", title=y_label, axis=alt.Axis(labelColor="#148F77"))
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

# Streamlit App
st.set_page_config(page_title="Token Feature Analysis", layout="wide", page_icon="🔍")
st.markdown("<h1 style='color:#1F618D;text-align:center;'>Token Feature Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Input with Submit Button
st.sidebar.markdown("<h3 style='color:#1ABC9C;'>Input Sentence</h3>", unsafe_allow_html=True)
sentence = st.sidebar.text_area("Enter a sentence:")
if st.sidebar.button("Generate Tokens"):
    st.session_state.tokens = tokenize_sentence(sentence)

# Tokenization and Features
if sentence:
    st.markdown("<h2 style='color:#1F618D;'>Sentence Tokenization</h2>", unsafe_allow_html=True)

    # CSS for uniform button styling and gap
    button_style = """
        <style>
        .stButton button {
            background-color: #F9EBEA;
            color: #943126;
            border: 2px solid #CB4335;
            border-radius: 10px;
            font-size: 16px;
            height: 50px;
            margin: 2px;  /* Small gap */
            padding: 5px 15px;
        }
        .stButton button:hover {
            background-color: #FDEDEC;
            color: #641E16;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Generate tokens and display as buttons
    tokens = tokenize_sentence(sentence)
    token_container = st.container()
    token_cols = token_container.columns(len(tokens))

    for i, token in enumerate(tokens):
        with token_cols[i]:
            if st.button(token, key=f"token_{i}"):
                st.session_state.selected_token = token

    if st.session_state.selected_token:
        st.markdown(f"<h3 style='color:#1ABC9C;'>Features for Token: <u>{st.session_state.selected_token}</u></h3>", unsafe_allow_html=True)
        explanations = fetch_features_for_token(st.session_state.selected_token)

        if explanations:
            # Extract descriptions from explanations
            descriptions = [exp.get("explanations", [{}])[0].get("description", "No description available") for exp in explanations]
            selected_description = st.selectbox("Select a Feature Description:", descriptions)

            if selected_description:
                # Find the corresponding explanation
                selected_feature = next(
                    (exp for exp in explanations if exp.get("explanations", [{}])[0].get("description") == selected_description),
                    None
                )
                if selected_feature:
                    # Display Positive and Negative Logits
                    neg_str = selected_feature.get("neuron", {}).get("neg_str", [])
                    neg_values = selected_feature.get("neuron", {}).get("neg_values", [])
                    pos_str = selected_feature.get("neuron", {}).get("pos_str", [])
                    pos_values = selected_feature.get("neuron", {}).get("pos_values", [])

                    st.markdown("<h4 style='color:#1F618D;'>Logits Table</h4>", unsafe_allow_html=True)
                    if neg_str and neg_values:
                        st.write("### Negative Logits")
                        st.write(pd.DataFrame({"Word": neg_str, "Value": neg_values}))
                    if pos_str and pos_values:
                        st.write("### Positive Logits")
                        st.write(pd.DataFrame({"Word": pos_str, "Value": pos_values}))

                    # Histograms
                    freq_x = selected_feature.get("neuron", {}).get("freq_hist_data_bar_values", [])
                    freq_y = selected_feature.get("neuron", {}).get("freq_hist_data_bar_heights", [])
                    logits_x = selected_feature.get("neuron", {}).get("logits_hist_data_bar_values", [])
                    logits_y = selected_feature.get("neuron", {}).get("logits_hist_data_bar_heights", [])

                    if freq_x and freq_y:
                        st.markdown("<h4 style='color:#1F618D;'>Frequency Histogram</h4>", unsafe_allow_html=True)
                        st.altair_chart(plot_graph(freq_x, freq_y, "Frequency Histogram"), use_container_width=True)
                    if logits_x and logits_y:
                        st.markdown("<h4 style='color:#1F618D;'>Logits Histogram</h4>", unsafe_allow_html=True)
                        st.altair_chart(plot_graph(logits_x, logits_y, "Logits Histogram"), use_container_width=True)
        else:
            st.warning("No features found for the selected token. Try another token.")

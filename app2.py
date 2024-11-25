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
if "tokens" not in st.session_state:
    st.session_state.tokens = []


# Helper Functions
def tokenize_sentence(sentence):
    """Tokenize the input sentence using regex."""
    return re.findall(r"\b\w+\b|[^\w\s]", sentence)


def fetch_explanations_for_token(token):
    """Fetch explanations for a given token using the Neuronpedia 'search-all' API."""
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
        result = response.json().get("result", [])

        if not result:
            st.warning(f"No explanations found for token: {token}")
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []


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
            margin: 5px;
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
        features = fetch_explanations_for_token(st.session_state.selected_token)

        if features:
            st.markdown("### Token Explanations")
            # Safely extract descriptions (now correctly extracting the description field)
            descriptions = [f.get("description", "No description available") for f in features]

            # Warn if all descriptions are missing
            if all(desc == "No description available" for desc in descriptions):
                st.warning("Descriptions are missing for all features of the selected token.")

            # Feature selection dropdown
            selected_description = st.selectbox("Select a Feature Description:", descriptions)

            if selected_description and selected_description != "No description available":
                # Find the selected feature by description
                selected_feature = next((f for f in features if f.get("description") == selected_description), None)
                if selected_feature:
                    neuron_data = selected_feature.get("neuron", {})
                    if neuron_data:
                        # Display Neuron Data
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown("### Negative Logits")
                            neg_str = neuron_data.get("neg_str", [])
                            neg_values = neuron_data.get("neg_values", [])
                            if neg_str and neg_values:
                                st.dataframe(pd.DataFrame({"Word": neg_str, "Value": neg_values}))
                            else:
                                st.write("No Negative Logits available.")
                        with cols[1]:
                            st.markdown("### Positive Logits")
                            pos_str = neuron_data.get("pos_str", [])
                            pos_values = neuron_data.get("pos_values", [])
                            if pos_str and pos_values:
                                st.dataframe(pd.DataFrame({"Word": pos_str, "Value": pos_values}))
                            else:
                                st.write("No Positive Logits available.")
                        
                        # Display Histograms
                        freq_x = neuron_data.get("freq_hist_data_bar_values", [])
                        freq_y = neuron_data.get("freq_hist_data_bar_heights", [])
                        if freq_x and freq_y:
                            st.markdown("### Frequency Histogram")
                            st.altair_chart(plot_graph(freq_x, freq_y, "Frequency Histogram", "Values", "Frequency"), use_container_width=True)
                        logits_x = neuron_data.get("logits_hist_data_bar_values", [])
                        logits_y = neuron_data.get("logits_hist_data_bar_heights", [])
                        if logits_x and logits_y:
                            st.markdown("### Logits Histogram")
                            st.altair_chart(plot_graph(logits_x, logits_y, "Logits Histogram", "Values", "Logits"), use_container_width=True)
                    else:
                        st.warning("No neuron data available for the selected feature.")
        else:
            st.warning("No features found for the selected token.")

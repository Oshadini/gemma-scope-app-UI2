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
        result = response.json()
        
        # Debugging: Display raw API response
        #st.write("API Response Debugging:", result)
        
        return result.get("result", [])
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

def plot_graph(x_data, y_data, title, x_label="X-axis", y_label="Y-axis"):
    """Generate a bar chart for visualization."""
    if not x_data or not y_data:
        return None
    chart = alt.Chart(pd.DataFrame({"x": x_data, "y": y_data})).mark_bar(color="#A3E4D7").encode(
        x=alt.X("x:Q", title=x_label),
        y=alt.Y("y:Q", title=y_label)
    ).properties(title=title, width=600, height=400)
    return chart

# Streamlit App
st.set_page_config(page_title="Token Feature Analysis", layout="wide", page_icon="🔍")
st.markdown("<h1 style='color:#1F618D;text-align:center;'>Token Feature Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Input
st.sidebar.markdown("<h3 style='color:#1ABC9C;'>Input Sentence</h3>", unsafe_allow_html=True)
sentence = st.sidebar.text_area("Enter a sentence:")
if st.sidebar.button("Generate Tokens"):
    st.session_state.tokens = tokenize_sentence(sentence)

# Token Buttons and Features
if st.session_state.tokens:
    st.markdown("<h2 style='color:#1F618D;'>Tokenized Sentence</h2>", unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.tokens))

    for idx, token in enumerate(st.session_state.tokens):
        with cols[idx]:
            if st.button(token, key=f"token_{idx}"):
                st.session_state.selected_token = token

# Display Features
if st.session_state.selected_token:
    st.markdown(f"<h3 style='color:#1ABC9C;'>Features for Token: {st.session_state.selected_token}</h3>", unsafe_allow_html=True)
    features = fetch_explanations_for_token(st.session_state.selected_token)

    if features:
        descriptions = [f.get("description", "No description available") for f in features]
        
        if not descriptions or all(desc == "No description available" for desc in descriptions):
            st.warning("No descriptions available for the selected token.")
        else:
            selected_desc = st.selectbox("Select a Feature Description:", descriptions)

            if selected_desc and selected_desc != "No description available":
                selected_feature = next((f for f in features if f.get("description") == selected_desc), None)
                if selected_feature:
                    neuron_data = selected_feature.get("neuron", {})
                    
                    # Display Logits and Histograms
                    if neuron_data:
                        st.markdown("### Neuron Data")
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown("#### Negative Logits")
                            neg_str = neuron_data.get("neg_str", [])
                            neg_values = neuron_data.get("neg_values", [])
                            if neg_str and neg_values:
                                st.dataframe(pd.DataFrame({"Word": neg_str, "Value": neg_values}))
                            else:
                                st.write("No Negative Logits available.")
                        with cols[1]:
                            st.markdown("#### Positive Logits")
                            pos_str = neuron_data.get("pos_str", [])
                            pos_values = neuron_data.get("pos_values", [])
                            if pos_str and pos_values:
                                st.dataframe(pd.DataFrame({"Word": pos_str, "Value": pos_values}))
                            else:
                                st.write("No Positive Logits available.")
                        
                        # Histograms
                        freq_x = neuron_data.get("freq_hist_data_bar_values", [])
                        freq_y = neuron_data.get("freq_hist_data_bar_heights", [])
                        if freq_x and freq_y:
                            st.altair_chart(plot_graph(freq_x, freq_y, "Frequency Histogram", "Values", "Frequency"), use_container_width=True)
                        logits_x = neuron_data.get("logits_hist_data_bar_values", [])
                        logits_y = neuron_data.get("logits_hist_data_bar_heights", [])
                        if logits_x and logits_y:
                            st.altair_chart(plot_graph(logits_x, logits_y, "Logits Histogram", "Values", "Logits"), use_container_width=True)
    else:
        st.warning("No features found for the selected token.")

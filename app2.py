# File: sentence_token_features_ui_with_advanced_ui.py

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
    "X-Api-Key": "sk-np-h0ZsR5M1gY0w8al332rJUYa0C8hQL2yUogd5n4Pgvvg0"  # Replace with your actual API token
}

# Initialize Session State
if "selected_token" not in st.session_state:
    st.session_state.selected_token = None
if "tokens" not in st.session_state:
    st.session_state.tokens = []
if "available_explanations" not in st.session_state:
    st.session_state.available_explanations = []

# Helper Functions
def tokenize_sentence(sentence):
    """Tokenize the input sentence using regex."""
    return re.findall(r"\b\w+\b|[^\w\s]", sentence)

def fetch_features_for_token(token):
    """Fetch features for a given token using the API."""
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
        
        # Debugging: Display the raw API response in Streamlit
        st.write("API Response Debugging:", result)
        
        explanations = result.get("results", [])
        if not explanations:
            st.warning("No features returned by the API.")
        return explanations
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

# Display Features with Enhanced Debugging
if st.session_state.selected_token:
    st.markdown(f"<h3 style='color:#1ABC9C;'>Features for Token: {st.session_state.selected_token}</h3>", unsafe_allow_html=True)
    features = fetch_features_for_token(st.session_state.selected_token)

    if features:
        # Extract explanations safely
        descriptions = [
            exp.get("explanations", [{}])[0].get("description", "No description available")
            for exp in features
            if exp.get("explanations")
        ]

        if descriptions:
            selected_desc = st.selectbox("Select a Feature Description:", descriptions)
            
            # Display additional debugging if no valid description is found
            if not any(desc != "No description available" for desc in descriptions):
                st.warning("Descriptions are missing or unavailable for the features.")

            if selected_desc and selected_desc != "No description available":
                feature = next(
                    (f for f in features if f.get("explanations", [{}])[0].get("description") == selected_desc),
                    None
                )
                if feature:
                    # Display logits and histograms (same as before)
                    neg_str = feature.get("neuron", {}).get("neg_str", [])
                    neg_values = feature.get("neuron", {}).get("neg_values", [])
                    pos_str = feature.get("neuron", {}).get("pos_str", [])
                    pos_values = feature.get("neuron", {}).get("pos_values", [])

                    if neg_str and neg_values:
                        st.write("### Negative Logits")
                        st.dataframe(pd.DataFrame({"Word": neg_str, "Value": neg_values}))

                    if pos_str and pos_values:
                        st.write("### Positive Logits")
                        st.dataframe(pd.DataFrame({"Word": pos_str, "Value": pos_values}))

                    freq_x = feature.get("neuron", {}).get("freq_hist_data_bar_values", [])
                    freq_y = feature.get("neuron", {}).get("freq_hist_data_bar_heights", [])
                    logits_x = feature.get("neuron", {}).get("logits_hist_data_bar_values", [])
                    logits_y = feature.get("neuron", {}).get("logits_hist_data_bar_heights", [])

                    if freq_x and freq_y:
                        st.altair_chart(plot_graph(freq_x, freq_y, "Frequency Histogram"), use_container_width=True)
                    if logits_x and logits_y:
                        st.altair_chart(plot_graph(logits_x, logits_y, "Logits Histogram"), use_container_width=True)
        else:
            st.warning("No valid descriptions found in the API response.")
    else:
        st.warning("No features found for the selected token.")

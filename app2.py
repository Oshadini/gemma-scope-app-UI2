# File: sentence_token_features_ui_with_advanced_ui.py

import streamlit as st
import requests
import pandas as pd
import altair as alt

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/explanation/search-model"
MODEL_ID = "gemma-2-9b-it"
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": "sk-np-h0ZsR5M1gY0w8al332rJUYa0C8hQL2yUogd5n4Pgvvg0"
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
    """Tokenize the input sentence."""
    return sentence.split()

def fetch_explanations_for_token(token):
    """Fetch explanations from Neuronpedia API for a given token."""
    payload = {
        "modelId": MODEL_ID,
        "query": token
    }
    try:
        response = requests.post(NEURONPEDIA_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        explanations = response.json().get("results", [])
        if not explanations:
            st.warning(f"No explanations found for token: {token}")
        return explanations
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

def plot_graph(x_data, y_data, title, x_label="X-axis", y_label="Y-axis"):
    """Generate a histogram for visualization."""
    if not x_data or not y_data:
        st.write(f"No data available for {title}.")
        return None
    chart = alt.Chart(pd.DataFrame({"x": x_data, "y": y_data})).mark_bar().encode(
        x=alt.X("x:Q", title=x_label),
        y=alt.Y("y:Q", title=y_label)
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

# Streamlit App
st.set_page_config(page_title="Token Feature Analysis", layout="wide")
st.title("Token Feature Analysis Dashboard")

# Sentence Input
st.sidebar.header("Input Sentence")
sentence = st.sidebar.text_area("Enter a sentence:", "Streamlit makes creating dashboards simple and intuitive!")

# Tokenization and Token Features
st.header("Sentence Tokenization and Features")
if sentence:
    tokens = tokenize_sentence(sentence)
    st.subheader("Select a Token")
    cols = st.columns(len(tokens))
    
    for i, token in enumerate(tokens):
        with cols[i]:
            if st.button(token):
                st.session_state.selected_token = token

    if st.session_state.selected_token:
        st.subheader(f"Fetching Features for Token: {st.session_state.selected_token}")

        # Fetch features from Neuronpedia API
        explanations = fetch_explanations_for_token(st.session_state.selected_token)

        if explanations:
            # Store explanations in session state
            st.session_state.available_explanations = explanations

            # Extract and display descriptions in a dropdown
            descriptions = [exp["description"] for exp in explanations]
            selected_description = st.selectbox("Select a Feature Description:", descriptions)

            if selected_description:
                # Find the explanation details corresponding to the selected description
                selected_feature = next((exp for exp in explanations if exp["description"] == selected_description), None)
                if selected_feature:
                    st.session_state.selected_explanation = selected_feature

                    # Access neuron data
                    neuron_data = selected_feature.get("neuron", {})
                    if not neuron_data:
                        st.warning("No neuron data available for the selected feature.")
                    else:
                        # Display Negative Logits
                        neg_str = neuron_data.get("neg_str", [])
                        neg_values = neuron_data.get("neg_values", [])
                        if neg_str and neg_values:
                            st.write("### Negative Logits")
                            st.write(pd.DataFrame({"Word": neg_str, "Value": neg_values}))
                        else:
                            st.write("No Negative Logits available.")

                        # Display Positive Logits
                        pos_str = neuron_data.get("pos_str", [])
                        pos_values = neuron_data.get("pos_values", [])
                        if pos_str and pos_values:
                            st.write("### Positive Logits")
                            st.write(pd.DataFrame({"Word": pos_str, "Value": pos_values}))
                        else:
                            st.write("No Positive Logits available.")

                        # Display Histogram: Frequency Data
                        freq_x = neuron_data.get("freq_hist_data_bar_values", [])  # Swapped X-axis
                        freq_y = neuron_data.get("freq_hist_data_bar_heights", [])  # Swapped Y-axis
                        if freq_x and freq_y:
                            st.write("### Frequency Histogram")
                            st.altair_chart(
                                plot_graph(freq_x, freq_y, "Frequency Histogram", x_label="Bar Values", y_label="Bar Heights"),
                                use_container_width=True
                            )
                        else:
                            st.write("No Frequency Histogram data available.")

                        # Display Histogram: Logits Data
                        logits_x = neuron_data.get("logits_hist_data_bar_values", [])  # Swapped X-axis
                        logits_y = neuron_data.get("logits_hist_data_bar_heights", [])  # Swapped Y-axis
                        if logits_x and logits_y:
                            st.write("### Logits Histogram")
                            st.altair_chart(
                                plot_graph(logits_x, logits_y, "Logits Histogram", x_label="Bar Values", y_label="Bar Heights"),
                                use_container_width=True
                            )
                        else:
                            st.write("No Logits Histogram data available.")
        else:
            st.warning("No features found for the selected token. Please try another token.")

# Footer
st.markdown("---")
st.text("Created by Streamlit - Example integrated with Neuronpedia API.")

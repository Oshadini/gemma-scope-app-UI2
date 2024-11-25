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
        response.raise_for_status()
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
    chart = alt.Chart(pd.DataFrame({"x": x_data, "y": y_data})).mark_bar(color="#FF5733").encode(
        x=alt.X("x:Q", title=x_label, axis=alt.Axis(labelColor="#2ECC71")),
        y=alt.Y("y:Q", title=y_label, axis=alt.Axis(labelColor="#3498DB"))
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

# Streamlit App
st.set_page_config(page_title="Token Feature Analysis", layout="wide", page_icon="🔍")
st.markdown("<h1 style='color:#3498DB;text-align:center;'>Token Feature Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Input
st.sidebar.markdown("<h3 style='color:#2ECC71;'>Input Sentence</h3>", unsafe_allow_html=True)
sentence = st.sidebar.text_area("Enter a sentence:", "Streamlit makes creating dashboards simple and intuitive!")

# Tokenization and Features
if sentence:
    st.markdown("<h2 style='color:#3498DB;'>Sentence Tokenization</h2>", unsafe_allow_html=True)
    tokens = tokenize_sentence(sentence)
    
    # Display tokens as buttons
    token_container = st.container()
    token_cols = token_container.columns(len(tokens))

    for i, token in enumerate(tokens):
        with token_cols[i]:
            if st.button(f"🔹 {token}", key=f"token_{i}"):
                st.session_state.selected_token = token

    if st.session_state.selected_token:
        st.markdown(f"<h3 style='color:#2ECC71;'>Features for Token: <u>{st.session_state.selected_token}</u></h3>", unsafe_allow_html=True)
        explanations = fetch_explanations_for_token(st.session_state.selected_token)

        if explanations:
            descriptions = [exp["description"] for exp in explanations]
            selected_description = st.selectbox("Select a Feature Description:", descriptions)

            if selected_description:
                selected_feature = next((exp for exp in explanations if exp["description"] == selected_description), None)
                if selected_feature:
                    neuron_data = selected_feature.get("neuron", {})
                    if not neuron_data:
                        st.warning("No neuron data available for the selected feature.")
                    else:
                        # Negative Logits
                        neg_str = neuron_data.get("neg_str", [])
                        neg_values = neuron_data.get("neg_values", [])
                        if neg_str and neg_values:
                            st.markdown("### Negative Logits")
                            neg_df = pd.DataFrame({"Word": neg_str, "Value": neg_values})
                            st.dataframe(neg_df.style.set_properties(
                                **{'background-color': '#FFC300', 'color': 'black'}))

                        # Positive Logits
                        pos_str = neuron_data.get("pos_str", [])
                        pos_values = neuron_data.get("pos_values", [])
                        if pos_str and pos_values:
                            st.markdown("### Positive Logits")
                            pos_df = pd.DataFrame({"Word": pos_str, "Value": pos_values})
                            st.dataframe(pos_df.style.set_properties(
                                **{'background-color': '#2ECC71', 'color': 'white'}))

                        # Frequency Histogram
                        freq_x = neuron_data.get("freq_hist_data_bar_values", [])
                        freq_y = neuron_data.get("freq_hist_data_bar_heights", [])
                        if freq_x and freq_y:
                            st.markdown("### Frequency Histogram")
                            st.altair_chart(
                                plot_graph(freq_x, freq_y, "Frequency Histogram", "Values", "Frequency"),
                                use_container_width=True
                            )

                        # Logits Histogram
                        logits_x = neuron_data.get("logits_hist_data_bar_values", [])
                        logits_y = neuron_data.get("logits_hist_data_bar_heights", [])
                        if logits_x and logits_y:
                            st.markdown("### Logits Histogram")
                            st.altair_chart(
                                plot_graph(logits_x, logits_y, "Logits Histogram", "Values", "Logits"),
                                use_container_width=True
                            )
        else:
            st.warning("No features found for the selected token. Try another token.")



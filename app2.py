# File: sentence_token_features_ui_with_api.py

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

HTML_TEMPLATE = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"


def get_dashboard_html(sae_release="gemma-2-9b", sae_id="20-gemmascope-res-16k", feature_idx=0):
    """Generate the iframe URL for the Neuronpedia dashboard."""
    return HTML_TEMPLATE.format(sae_release, sae_id, feature_idx)


# Initialize Session State
if "selected_token" not in st.session_state:
    st.session_state.selected_token = None
if "available_explanations" not in st.session_state:
    st.session_state.available_explanations = []
if "selected_explanation" not in st.session_state:
    st.session_state.selected_explanation = {}
if "api_response" not in st.session_state:
    st.session_state.api_response = {}


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
        st.session_state.api_response = response.json()  # Store the API response in session state
        explanations = st.session_state.api_response.get("explanations", [])
        if not explanations:
            st.warning(f"No explanations found for token: {token}")
        return explanations
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []


def plot_graph(x_data, y_data, title, x_label="X-axis", y_label="Y-axis"):
    """Generate a histogram for visualization."""
    chart = alt.Chart(pd.DataFrame({"x": x_data, "y": y_data})).mark_bar().encode(
        x=alt.X("x:Q", title=x_label),
        y=alt.Y("y:Q", title=y_label)
    ).properties(
        title=title,
        width=400,
        height=300
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
    st.session_state.selected_token = st.radio("Tokens in Sentence:", tokens, horizontal=True)

    if st.session_state.selected_token:
        st.subheader(f"Fetching Features for Token: {st.session_state.selected_token}")

        # Fetch features from Neuronpedia API
        explanations = fetch_explanations_for_token(st.session_state.selected_token)

        # Display raw API response in an expandable section
        with st.expander("View API Response"):
            st.json(st.session_state.api_response)

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

                    # Display feature details
                    st.write(f"Details for `{selected_description}`:")

                    # Display Negative Logits
                    neg_str = selected_feature.get("neg_str", [])
                    neg_values = selected_feature.get("neg_values", [])
                    if neg_str and neg_values:
                        st.write("### Negative Logits")
                        for word, value in zip(neg_str, neg_values):
                            st.write(f"{word}: {value}")

                    # Display Positive Logits
                    pos_str = selected_feature.get("pos_str", [])
                    pos_values = selected_feature.get("pos_values", [])
                    if pos_str and pos_values:
                        st.write("### Positive Logits")
                        for word, value in zip(pos_str, pos_values):
                            st.write(f"{word}: {value}")

                    # Display Histogram: Frequency Data
                    freq_x = selected_feature.get("freq_hist_data_bar_heights", [])
                    freq_y = selected_feature.get("freq_hist_data_bar_values", [])
                    if freq_x and freq_y:
                        st.write("### Frequency Histogram")
                        st.altair_chart(plot_graph(freq_x, freq_y, "Frequency Histogram", x_label="Bar Heights", y_label="Bar Values"), use_container_width=True)

                    # Display Histogram: Logits Data
                    logits_x = selected_feature.get("logits_hist_data_bar_heights", [])
                    logits_y = selected_feature.get("logits_hist_data_bar_values", [])
                    if logits_x and logits_y:
                        st.write("### Logits Histogram")
                        st.altair_chart(plot_graph(logits_x, logits_y, "Logits Histogram", x_label="Bar Heights", y_label="Bar Values"), use_container_width=True)

                    # Embed Neuronpedia Dashboard
                    feature_idx = selected_feature.get("index", 0)
                    sae_id = "20-gemmascope-res-16k"  # Example SAE ID
                    html = get_dashboard_html(sae_release="gemma-2-9b", sae_id=sae_id, feature_idx=feature_idx)
                    st.markdown(f'<iframe src="{html}" width="1200" height="600"></iframe>', unsafe_allow_html=True)

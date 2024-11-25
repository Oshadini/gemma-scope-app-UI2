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
        .token-button {
            background-color: #E8F8F5;
            color: #0E6655;
            border: 1px solid #117A65;
            padding: 10px 15px;
            margin: 5px;
            text-align: center;
            font-size: 14px;
            border-radius: 5px;
        }
        .token-button:hover {
            background-color: #D1F2EB;
            color: #145A32;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    # Generate tokens and display as styled buttons
    tokens = tokenize_sentence(sentence)
    token_container = st.container()
    token_cols = token_container.columns(len(tokens))

    for i, token in enumerate(tokens):
        with token_cols[i]:
            if st.markdown(f"<div class='token-button'>{token}</div>", unsafe_allow_html=True):
                st.session_state.selected_token = token

    if st.session_state.selected_token:
        st.markdown(f"<h3 style='color:#1ABC9C;'>Features for Token: <u>{st.session_state.selected_token}</u></h3>", unsafe_allow_html=True)
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
                        # Logits Tables with Light Background Colors
                        st.markdown("<h4 style='color:#1F618D;'>Logits Tables</h4>", unsafe_allow_html=True)
                        cols = st.columns(2)

                        with cols[0]:
                            st.markdown("### Negative Logits")
                            neg_str = neuron_data.get("neg_str", [])
                            neg_values = neuron_data.get("neg_values", [])
                            if neg_str and neg_values:
                                neg_df = pd.DataFrame({"Word": neg_str, "Value": neg_values})
                                st.dataframe(neg_df.style.set_properties(
                                    **{'background-color': '#FDEDEC', 'color': '#A93226'}))
                            else:
                                st.write("No Negative Logits available.")

                        with cols[1]:
                            st.markdown("### Positive Logits")
                            pos_str = neuron_data.get("pos_str", [])
                            pos_values = neuron_data.get("pos_values", [])
                            if pos_str and pos_values:
                                pos_df = pd.DataFrame({"Word": pos_str, "Value": pos_values})
                                st.dataframe(pos_df.style.set_properties(
                                    **{'background-color': '#EBF5FB', 'color': '#1A5276'}))
                            else:
                                st.write("No Positive Logits available.")

                        # Frequency Histogram
                        freq_x = neuron_data.get("freq_hist_data_bar_values", [])
                        freq_y = neuron_data.get("freq_hist_data_bar_heights", [])
                        if freq_x and freq_y:
                            st.markdown("<h4 style='color:#1F618D;'>Frequency Histogram</h4>", unsafe_allow_html=True)
                            st.altair_chart(
                                plot_graph(freq_x, freq_y, "Frequency Histogram", "Values", "Frequency"),
                                use_container_width=True
                            )

                        # Logits Histogram
                        logits_x = neuron_data.get("logits_hist_data_bar_values", [])
                        logits_y = neuron_data.get("logits_hist_data_bar_heights", [])
                        if logits_x and logits_y:
                            st.markdown("<h4 style='color:#1F618D;'>Logits Histogram</h4>", unsafe_allow_html=True)
                            st.altair_chart(
                                plot_graph(logits_x, logits_y, "Logits Histogram", "Values", "Logits"),
                                use_container_width=True
                            )
        else:
            st.warning("No features found for the selected token. Try another token.")

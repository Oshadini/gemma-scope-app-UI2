# File: sentence_token_features_ui_with_api.py

import streamlit as st
import requests
import pandas as pd
import altair as alt

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/explanation/search-model"
MODEL_ID = "gemma-2-9b-it"
HEADERS = {"Content-Type": "application/json", "X-Api-Key": "YOUR_TOKEN"}  # Replace with your API token

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
        data = response.json()
        return data.get("explanations", [])
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
        if explanations:
            st.session_state.available_explanations = explanations
            explanation_options = {exp["description"]: exp for exp in explanations}
            selected_description = st.selectbox("Select a Feature:", list(explanation_options.keys()))
            
            if selected_description:
                st.session_state.selected_explanation = explanation_options[selected_description]
                selected_feature = st.session_state.selected_explanation

                # Display description details
                st.write(f"Details for `{selected_description}`:")
                
                # Display Negative Logits
                neg_str = selected_feature.get("neg_str", [])
                neg_values = selected_feature.get("neg_values", [])
                if neg_str and neg_values:
                    st.write("### Negative Logits")
                    st.write(f"Words: {', '.join(neg_str)}")
                    st.write(f"Values: {neg_values}")
                    st.altair_chart(plot_graph(range(len(neg_values)), neg_values, "Negative Logits"), use_container_width=True)

                # Display Positive Logits
                pos_str = selected_feature.get("pos_str", [])
                pos_values = selected_feature.get("pos_values", [])
                if pos_str and pos_values:
                    st.write("### Positive Logits")
                    st.write(f"Words: {', '.join(pos_str)}")
                    st.write(f"Values: {pos_values}")
                    st.altair_chart(plot_graph(range(len(pos_values)), pos_values, "Positive Logits"), use_container_width=True)

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
        else:
            st.warning("No features found for the selected token. Please try another token.")

# Footer
st.markdown("---")
st.text("Created by Streamlit - Example integrated with Neuronpedia API.")

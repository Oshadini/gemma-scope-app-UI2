# File: sentence_token_features_ui_with_api.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/explanation/search-model"
API_KEY = "YOUR_TOKEN"  # Replace with your actual Neuronpedia API token
MODEL_ID = "gemma-2-9b-it"

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
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": API_KEY
    }
    try:
        response = requests.post(NEURONPEDIA_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        data = response.json()
        explanations = data.get("explanations", [])
        return explanations
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

def plot_graph(data, title):
    """Generate a simple bar chart for visualization."""
    chart = alt.Chart(pd.DataFrame({"x": range(len(data)), "y": data})).mark_bar().encode(
        x="x",
        y="y"
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
    # Tokenize the sentence
    tokens = tokenize_sentence(sentence)
    selected_token = st.radio("Tokens in Sentence:", tokens, horizontal=True)
    
    if selected_token:
        st.subheader(f"Fetching Features for Token: {selected_token}")
        
        # Fetch features from Neuronpedia API
        explanations = fetch_explanations_for_token(selected_token)
        if explanations:
            # Populate dropdown with explanation descriptions
            explanation_options = {exp["description"]: exp for exp in explanations}
            selected_description = st.selectbox("Select a Feature:", list(explanation_options.keys()))
            
            if selected_description:
                st.write(f"Details for `{selected_description}`:")
                selected_feature = explanation_options[selected_description]

                # Display related graphs (dummy data for demonstration)
                pos_logits = np.random.rand(10)  # Replace with real data if available
                neg_logits = np.random.rand(10)  # Replace with real data if available
                activation_density = np.random.rand(10)  # Replace with real data if available
                
                st.altair_chart(plot_graph(pos_logits, "Positive Logits"), use_container_width=True)
                st.altair_chart(plot_graph(neg_logits, "Negative Logits"), use_container_width=True)
                st.altair_chart(plot_graph(activation_density, "Activation Density"), use_container_width=True)
        else:
            st.warning("No features found for the selected token. Please try another token.")

# Footer
st.markdown("---")
st.text("Created by Streamlit - Example integrated with Neuronpedia API.")

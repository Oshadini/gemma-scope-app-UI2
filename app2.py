# File: sentence_token_features_ui.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Dummy functions to simulate data for features and graphs
def tokenize_sentence(sentence):
    """Tokenize the input sentence."""
    return sentence.split()

def get_token_features(token):
    """Return dummy features for a given token."""
    features = {
        "Feature A": {"pos_logits": np.random.rand(10), "neg_logits": np.random.rand(10), "activation_density": np.random.rand(10)},
        "Feature B": {"pos_logits": np.random.rand(10), "neg_logits": np.random.rand(10), "activation_density": np.random.rand(10)},
        "Feature C": {"pos_logits": np.random.rand(10), "neg_logits": np.random.rand(10), "activation_density": np.random.rand(10)},
    }
    return features

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
        st.subheader(f"Features for Token: {selected_token}")
        
        # Display features for the selected token
        features = get_token_features(selected_token)
        feature_keys = list(features.keys())
        selected_feature = st.selectbox("Select a Feature:", feature_keys)
        
        if selected_feature:
            st.write(f"Visualizing data for `{selected_feature}`")
            
            # Display related graphs
            pos_logits = features[selected_feature]["pos_logits"]
            neg_logits = features[selected_feature]["neg_logits"]
            activation_density = features[selected_feature]["activation_density"]
            
            st.altair_chart(plot_graph(pos_logits, "Positive Logits"), use_container_width=True)
            st.altair_chart(plot_graph(neg_logits, "Negative Logits"), use_container_width=True)
            st.altair_chart(plot_graph(activation_density, "Activation Density"), use_container_width=True)

# Footer
st.markdown("---")
st.text("Created by Streamlit - Example inspired by BA and BB images.")

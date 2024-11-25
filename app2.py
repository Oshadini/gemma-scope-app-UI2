import streamlit as st
import requests

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/search-all"
MODEL_ID = "gpt2-small"
SOURCE_SET = "res-jb"
SELECTED_LAYERS = ["6-res-jb"]
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": "YOUR_TOKEN"  # Replace with your actual API token
}

# Helper Function to Fetch API Response
def fetch_api_response_for_text(text):
    """Fetch and display the raw API response for a given text."""
    payload = {
        "modelId": MODEL_ID,
        "sourceSet": SOURCE_SET,
        "text": text,
        "selectedLayers": SELECTED_LAYERS,
        "sortIndexes": [1],
        "ignoreBos": False,
        "densityThreshold": -1,
        "numResults": 50,
    }
    try:
        response = requests.post(NEURONPEDIA_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()  # Return the parsed JSON response
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Streamlit App
st.set_page_config(page_title="API Response Viewer", layout="wide")
st.markdown("<h1 style='color:#1F618D;text-align:center;'>Neuronpedia API Response Viewer</h1>", unsafe_allow_html=True)

# Sidebar Input
text_input = st.sidebar.text_input("Enter text to fetch API response:", "frog")
if st.sidebar.button("Fetch API Response"):
    with st.spinner("Fetching API response..."):
        response_data = fetch_api_response_for_text(text_input)
    
    # Display the raw response
    if "error" in response_data:
        st.error(f"Error: {response_data['error']}")
    else:
        st.json(response_data)  # Display the JSON response in an interactive viewer

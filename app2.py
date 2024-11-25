import requests

# Constants for the API
url = "https://www.neuronpedia.org/api/search-all"
headers = {
    "Content-Type": "application/json",
    "X-Api-Key": "YOUR_TOKEN"  # Replace with your actual API token
}

# Update the payload with "frog" as text
payload = {
    "modelId": "gpt2-small",
    "sourceSet": "res-jb",
    "text": "frog",  # Pass "frog" here
    "selectedLayers": ["6-res-jb"],
    "sortIndexes": [1],
    "ignoreBos": False,
    "densityThreshold": -1,
    "numResults": 50
}

# Send the POST request
try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes
    # Print the raw JSON response
    print("API Response:", response.json())
except requests.exceptions.RequestException as e:
    print(f"API Error: {e}")

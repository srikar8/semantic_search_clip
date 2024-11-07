import os
# Set OpenMP environment variables before importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import faiss
import streamlit as st
import clip
import torch
from PIL import Image
import json
import numpy as np
import requests
from io import BytesIO

import pandas as pd


# Load the JSON file
with open('coco_dataset/annotations/captions_train2017.json', 'r') as file:
    data = json.load(file)
# Assuming 'annotations' contains the captions, print the first entry
annotations = data['images']
annotat_df = pd.DataFrame(annotations)

def is_valid_image_url(image_url):
    try:
        # Send a request to the URL
        response = requests.get(image_url)
        
        # Check if the request was successful
        if response.status_code != 200:
            return False
        
        # Attempt to open the image
        img = Image.open(BytesIO(response.content))
        img.verify()  # Verify that the file is an actual image
        return True
    except (IOError, SyntaxError, requests.exceptions.RequestException):
        return False

# Load the pre-trained CLIP model
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load FAISS indexes and metadata
text_index = faiss.read_index("https://drive.google.com/file/d/1-joAEJ8KDNW4uzY5j7slj7cXfMLZU0b0/view?usp=share_link")
image_index = faiss.read_index("https://drive.google.com/file/d/1N3NSSA9xfJNqJb_2d3CQUt0KrFZJMEBv/view?usp=share_link")

with open("https://drive.google.com/file/d/1VbYbzDYpRG-5YkRSinMzLgOYIFPgkNhE/view?usp=share_link", "r") as f:
    image_metadata = json.load(f)

with open("https://drive.google.com/file/d/1rjdHlJSZcM5_EecD_PU9-LlFSflc6QpN/view?usp=share_link", "r") as f:
    text_metadata = json.load(f)

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Function to encode text
def encode_text(text):
    text_tokenized = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokenized)
    text_features = text_features.cpu().numpy()
    text_features = normalize_vectors(text_features)
    return text_features

# Function to encode image
def encode_image(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features = image_features.cpu().numpy()
    image_features = normalize_vectors(image_features)
    return image_features

# Function to search FAISS index
def search_index(index, query_vector,metadata, k=30):
    # FAISS inner product search
    similarity_scores, indices = index.search(query_vector, k)
    results = []
    seen_metadata = set()
    for i in range(k):
        idx = indices[0][i]
        score = similarity_scores[0][i]
        if idx < len(metadata):
            if metadata[idx]["image_id"] not in seen_metadata:
                seen_metadata.add(metadata[idx]["image_id"])
                results.append({
                    "metadata": metadata[idx],
                    "score": float(score)  # Convert from numpy.float32 to float
                    })
                
    
    return results  # List of results with metadata and similarity scores

def display_results(results, annotat_df, top_n=5):
    """
    Display the top search results with images, captions, and similarity scores.
    
    Parameters:
    - results (list): List of result dictionaries containing 'metadata' and 'score'.
    - annotat_df (DataFrame): DataFrame containing image metadata including URLs.
    - top_n (int): Number of top results to display (default is 5).
    """
    count = 0
    for result in results:
        image_id = result["metadata"]["image_id"]
        image_path = annotat_df.loc[annotat_df['id'] == image_id, 'flickr_url'].values[0]
        captions = result["metadata"]["caption"]
        score = result["score"]
        
        if is_valid_image_url(image_path):
            st.image(image_path, caption=f"{captions} (Similarity: {score:.4f})", use_column_width=True)
            count += 1
        
        if count >= top_n:
            break

# Title of the application
st.title("CLIP-Based Text and Image Search")

# Creating tabs for the search options
tab1, tab2 = st.tabs(["Search by Text", "Search by Image"])

# First tab: Search by Text
with tab1:
    st.subheader("Search by Text")
    text_query = st.text_input("Enter your text query:")
    
    # Display sample queries as clickable options
    sample_queries = [" woman is staring at a cell phone", "Delta passenger airplane", "A mountain landscape", " hot dog with fries"]
    st.write("### Sample Queries:")
    sample_triggered = False  # Track if a sample is clicked
    for query in sample_queries:
        if st.button(query, key=f"sample_text_{query}"):
            text_query = query
            sample_triggered = True  # Set flag if sample clicked

    # Trigger search if sample is selected or search button is clicked
    if (st.button("Search", key="text_search") or sample_triggered) and text_query:
        # Encode the text query and search for results
        text_features = encode_text(text_query)
        results = search_index(text_index, text_features, text_metadata)
        st.write("### Results:")
        display_results(results, annotat_df)

# Second tab: Search by Image
with tab2:
    st.subheader("Search by Image")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Display sample images as options
    st.write("### Sample Images:")
    sample_image_paths = ["sample_images/aeroplane.jpg", "sample_images/bikes.jpg", "sample_images/hotdog.jpg", "sample_images/phone.jpg","sample_images/eating.jpg"]
    sample_images = [Image.open(path) for path in sample_image_paths]
    sample_image_selected = None  # Track selected sample image
    
    cols = st.columns(len(sample_images))  # Create one column per image

    for idx, (col, sample_image) in enumerate(zip(cols, sample_images)):
        with col:
            st.image(sample_image, use_column_width=True)
            if st.button("Use this image", key=f"sample_image_{idx}"):
                sample_image_selected = sample_image

    # Use uploaded or sample image if available
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
    elif sample_image_selected:
        image = sample_image_selected
    else:
        image = None

    # Display and search if an image is available
    if image is not None:
        st.image(image, caption="Selected Image", use_column_width=True)
        
        # Trigger search immediately if sample image is selected or button is clicked
        if st.button("Search", key="image_search") or sample_image_selected:
            image_features = encode_image(image)
            results = search_index(image_index, image_features, image_metadata)
            st.write("### Results:")
            display_results(results, annotat_df)


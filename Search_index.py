import os
import json
import torch
from PIL import Image
import faiss
import numpy as np
from tqdm import tqdm
from srikar_clip_model import CLIP 

import albumentations as A
from albumentations.pytorch import ToTensorV2

preprocess = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ToTensorV2()
    ])

def load_coco_captions(annotations_file):
    """Load COCO captions from JSON file."""
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    return data

def process_coco_with_clip(annotations_data, images_dir, batch_size=32):
    """Process COCO images and captions with CLIP."""
    # Set device to MPS if available, else CPU
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    # Load CLIP model
    #model, preprocess = clip.load("ViT-B/32", device=device)
    
    model=CLIP()
    checkpoint = torch.load('model_save/checkpoint_20241109_164739.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Initialize lists to store embeddings and metadata
    image_embeddings = []
    text_embeddings = []
    metadata = []
    
    # Process in batches
    for i in tqdm(range(0, len(annotations_data['annotations']), batch_size)):
        batch_annotations = annotations_data['annotations'][i:i + batch_size]
        
        # Process images
        batch_images = []
        batch_captions = []
        batch_metadata = []
        
        for ann in batch_annotations:
            image_id = str(ann['image_id']).zfill(12)
            image_path = os.path.join(images_dir, f'{image_id}.jpg')
            
            try:
                image = Image.open(image_path)
                # Apply preprocessing and add a batch dimension (unsqueeze(0) for batch size 1)
                transformed = preprocess(image=np.array(image))
                # Convert to numpy array (optional, if you need it)
                image = transformed['image'].numpy()
                batch_images.append(torch.tensor(image).unsqueeze(0))
                batch_captions.append(ann['caption'])
                batch_metadata.append({
                    'image_id': ann['image_id'],
                    'caption': ann['caption']
                })
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Get CLIP embeddings
        with torch.no_grad():
            # Process images
            images_input = torch.cat(batch_images).to(device)
            image_features = model.encode_image(images_input)
            image_features = image_features.cpu().numpy()
            
            # Process text
            text_tokens = clip.tokenize(batch_captions).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy()
        
        # Normalize embeddings
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        
        # Store results
        image_embeddings.extend(image_features)
        text_embeddings.extend(text_features)
        metadata.extend(batch_metadata)
        
    
    return np.array(image_embeddings), np.array(text_embeddings), metadata

def create_faiss_index(embeddings, index_type="L2"):
    """Create FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    
    if index_type == "L2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "IP":  # Inner Product
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError("Unsupported index type")
    
    index.add(embeddings.astype('float32'))
    return index

def save_vectordb(index, metadata, output_dir):
    """Save FAISS index and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "vectors.index"))
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

def main():
    # Configuration
    annotations_file = "coco/annotations/captions_train2017.json"
    images_dir = "coco/images/train2017"
    output_dir = "clip_db_7"
    batch_size = 32  # Adjust based on your GPU/MPS memory
    
    # Load COCO annotations
    print("Loading COCO annotations...")
    annotations_data = load_coco_captions(annotations_file)
    
    # Process with CLIP
    print("Processing images and captions with CLIP...")
    image_embeddings, text_embeddings, metadata = process_coco_with_clip(
        annotations_data, images_dir, batch_size
    )
    
    # Create FAISS indices
    print("Creating FAISS indices...")
    image_index = create_faiss_index(image_embeddings)
    text_index = create_faiss_index(text_embeddings)
    
    # Save results
    print("Saving vector database...")
    save_vectordb(image_index, metadata, os.path.join(output_dir, "images"))
    save_vectordb(text_index, metadata, os.path.join(output_dir, "text"))
    
    print("Done! Vector database created successfully.")

if __name__ == "__main__":
    main()
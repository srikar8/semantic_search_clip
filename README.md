# Semantic Image Search Engine

A powerful semantic image search engine built using CLIP (Contrastive Language-Image Pre-training) and FAISS for efficient similarity search. This application allows users to search through a large collection of images using either text descriptions or similar images.

## Features

- Dual search modes:
  - Text-to-Image: Search images using natural language descriptions
  - Image-to-Image: Find similar images by uploading a reference image
- Interactive web interface built with Streamlit
- Efficient similarity search using FAISS indexing
- Sample queries and images for quick testing
- Support for both local images and Flickr URLs

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/srikar8/semantic_search_clip
cd semantic-image-search
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the COCO dataset:
- Create a directory named `coco_dataset`
- Download the COCO 2017 training images and annotations
- Place them in the following structure:
```
coco_dataset/
├── images/
│   └── train2017/
└── annotations/
    └── captions_train2017.json
```

4. Create a `sample_images` directory and add some sample images for the demo interface.

## Project Structure

```
├── app.py                 # Main Streamlit application
├── clip_model.py          # CLIP model implementation
├── Model_Training.ipynb   # Train CLIP model from scratch
├── search_index.py        # FAISS index creation and search functionality
├── requirements.txt       # Project dependencies
├── sample_images/         # Sample images for demo
└── clip_db_6/            # Directory for FAISS indexes
    ├── images/
    │   ├── vectors.index
    │   └── metadata.json
    └── text/
        ├── vectors.index
        └── metadata.json
```

## Usage

1. Build the search index:
```bash
python search_index.py
```

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Access the web interface at `http://localhost:8501`

### Search Options

1. Text Search:
   - Enter a natural language description
   - Use provided sample queries
   - View matching images with their captions

2. Image Search:
   - Upload an image file
   - Select from sample images
   - View similar images from the database

## Technical Details

### CLIP Model
- Uses a Vision Transformer (ViT) for image encoding
- Implements a Text Transformer for text encoding
- Trained on image-text pairs for semantic understanding

### Search Infrastructure
- FAISS indexes for both image and text vectors
- Normalized vectors for cosine similarity search
- Efficient batch processing for large datasets

### Performance Optimizations
- Environment variables for thread control
- Batch processing for efficient GPU utilization
- Image URL validation to handle broken links

## Environment Variables

The application sets several environment variables for optimal performance:
```python
KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- OpenAI CLIP model
- FAISS library by Facebook Research
- COCO dataset
- Streamlit framework
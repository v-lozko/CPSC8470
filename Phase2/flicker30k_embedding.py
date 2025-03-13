import torch
import numpy as np
import h5py
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load CLIP Model from ./models
print("Loading CLIP ViT model...")
model_path = "./models/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

#Load Flickr30K Dataset
print("Loading Flickr30K dataset from Hugging Face...")
dataset = load_dataset("nlphuji/flickr30k")["test"]

#Generate Image Embeddings
BATCH_SIZE = 256 # Process images in batches

def get_clip_embeddings(images):
    """Computes embeddings for a list of images using CLIP ViT in batches."""
    embeddings = []

    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Processing batches"):
        batch_images = images[i:i + BATCH_SIZE]  # Get batch

        # Convert images to PIL and preprocess
        pil_images = [img.convert("RGB") for img in batch_images]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)

        # Compute batch embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Normalize embeddings for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Append batch results
        embeddings.append(image_features.cpu().numpy())

    return np.vstack(embeddings)  # Shape: (num_images, embedding_dim)

print("Generating image embeddings in batches...")
image_embeddings = get_clip_embeddings(dataset["image"])
print("Embeddings shape:", image_embeddings.shape)


#Define Queries (Subset of Data)
query_ratio = 0.20  # Use 20% of dataset as queries
num_queries = int(query_ratio * len(image_embeddings))

# Randomly select a subset as queries
np.random.seed(42)
query_indices = np.random.choice(len(image_embeddings), num_queries, replace=False)

query_embeddings = image_embeddings[query_indices]
image_embeddings = np.delete(image_embeddings, query_indices, axis=0)
faiss.normalize_L2(image_embeddings)
faiss.normalize_L2(query_embeddings)

#Compute Nearest Neighbors for Queries Only
print(f"Computing nearest neighbors for {num_queries} queries using FAISS...")
d = image_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatIP(d)  # Inner Product (dot product ≈ cosine similarity)
index.add(image_embeddings)

# Perform nearest neighbor search for queries only
top_k = 1
D, I = index.search(query_embeddings, top_k)  # `I` contains indices of nearest neighbors

#Split Queries into Train (60%), Validation (20%), and Test (20%)
#split queries into train (60%) and remaining (40%)
train_queries, remaining_queries, train_idx, remaining_idx = train_test_split(
    query_embeddings, np.arange(len(query_embeddings)), test_size=0.4, random_state=42
)

#Split remaining into validation (20%) and test (20%)
valid_queries, test_queries, valid_idx, test_idx = train_test_split(
    remaining_queries, remaining_idx, test_size=0.5, random_state=42
)

#Assign nearest neighbors correctly
train_neighbors = I[train_idx]
valid_neighbors = I[valid_idx]
test_neighbors = I[test_idx]

#Save Queries & Nearest Neighbors to HDF5
hdf5_path = "./mips-learnt-ivf-main/Data/flickr30k_embeddings.hdf5"
print(f"Saving query embeddings and neighbors to {hdf5_path}...")

with h5py.File(hdf5_path, "w") as f:
    f.create_dataset("images", data=image_embeddings)  # Store full image embeddings
    f.create_dataset("train_queries", data=train_queries)
    f.create_dataset("valid_queries", data=valid_queries)
    f.create_dataset("test_queries", data=test_queries)

    f.create_dataset("train_neighbors", data=train_neighbors, dtype="int32")
    f.create_dataset("valid_neighbors", data=valid_neighbors, dtype="int32")
    f.create_dataset("test_neighbors", data=test_neighbors, dtype="int32")

print("Dataset successfully saved to HDF5!")

#Verify Saved Data
with h5py.File(hdf5_path, "r") as f:
    print("Available HDF5 keys:", list(f.keys()))
    print("Saved HDF5 keys:", list(f.keys()))  # Should contain "train_queries", "train_neighbors", etc.

print("All steps completed successfully! Your dataset is ready for main_mips.py")

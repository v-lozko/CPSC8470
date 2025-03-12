import torch
import numpy as np
import h5py
import faiss
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
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

#Image Preprocessing & Parameters
BATCH_SIZE = 256
image_folder = "./coco/train2017"  # Path where COCO images are stored

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for CLIP
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
])

#Generate Image Embeddings
BATCH_SIZE = 256 # Process images in batches

def get_clip_embeddings(image_folder):
    """Processes all images in a folder and computes CLIP embeddings."""
    embeddings = []
    filenames = sorted(os.listdir(image_folder))  # Ensure consistent order

    for i in tqdm(range(0, len(filenames), BATCH_SIZE), desc="Processing images"):
        batch_files = filenames[i:i + BATCH_SIZE]
        batch_images = []

        for file in batch_files:
            img_path = os.path.join(image_folder, file)
            image = Image.open(img_path).convert("RGB")  # Open image
            image = transform(image)  # Apply CLIP preprocessing
            batch_images.append(image)

        batch_images = torch.stack(batch_images).to(device)  # Convert to tensor

        # Compute embeddings
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=batch_images)

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())

    return np.vstack(embeddings)  # Return final embedding matrix

# Compute embeddings for all images
print("generating embeddings")
image_embeddings = get_clip_embeddings(image_folder)
print(f"Generated {len(image_embeddings)} embeddings for COCO.")

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
print("Splitting queries into train (60%), validation (20%), and test (20%)...")
# Step 1: First split queries into train (60%) and remaining (40%)
train_queries, remaining_queries, train_idx, remaining_idx = train_test_split(
    query_embeddings, np.arange(len(query_embeddings)), test_size=0.4, random_state=42
)

# Step 2: Split remaining into validation (20%) and test (20%)
valid_queries, test_queries, valid_idx, test_idx = train_test_split(
    remaining_queries, remaining_idx, test_size=0.5, random_state=42
)

# Step 3: Assign nearest neighbors correctly
train_neighbors = I[train_idx]
valid_neighbors = I[valid_idx]
test_neighbors = I[test_idx]

#Save Queries & Nearest Neighbors to HDF5
hdf5_path = "./mips-learnt-ivf-main/Data/ms_coco_embeddings.hdf5"
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

# Verify Saved Data
with h5py.File(hdf5_path, "r") as f:
    print("Saved HDF5 keys:", list(f.keys()))  # Should contain "train_queries", "train_neighbors", etc.

print("All steps completed successfully!")
from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import requests
import os
import json
import uuid
from zipfile import ZipFile
from threading import Thread
from more_itertools import chunked
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import sys



app = Flask(__name__)

# Load model & processor
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Auto device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load precomputed body shape clothing embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "body_shape_embeddings.pkl")

# Now open the file safely
with open(PKL_PATH, "rb") as f:
    body_shape_embedding_store = pickle.load(f)

# --- Utility ---
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

def tag_body_shapes_for_image(image_embedding, embedding_store, min_threshold=0.3, min_matches=1):
    body_shape_scores = {}
    for shape, clothing_dict in embedding_store.items():
        matches = 0
        for _, emb in clothing_dict.items():
            sim = cosine_similarity([image_embedding], [emb])[0][0]
            if sim >= min_threshold:
                matches += 1
        if matches >= min_matches:
            body_shape_scores[shape] = matches

    return sorted(body_shape_scores.keys(), key=lambda s: -body_shape_scores[s])


def normalize_embeddings(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)

def safe_vector(vec, dim=512):
    if not vec or len(vec) != dim:
        return [1.0 / (dim ** 0.5)] * dim
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else [1.0 / (dim ** 0.5)] * dim

# --- Text Embedding Endpoint ---
@app.route("/embed_text", methods=["POST"])
def embed_text():
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Missing or invalid 'texts' list"}), 400

    try:
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        emb_np = normalize_embeddings(embeddings.cpu().numpy())
        return jsonify({"embeddings": emb_np.tolist()})
    except Exception as e:
        return jsonify({"error": f"Failed to process text: {str(e)}"}), 500

# --- Image Embedding Endpoint ---
@app.route("/embed_images", methods=["POST"])
def embed_images():
    data = request.get_json()
    image_urls = data.get("image_urls", [])
    if not image_urls or not isinstance(image_urls, list):
        return jsonify({"error": "Missing or invalid 'image_urls' list"}), 400

    images = []
    valid_indices = []
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(response.raw).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"[embed_images] Failed to load image {url}: {e}")

    results = [[0.0] * 512 for _ in range(len(image_urls))]

    if images:
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            emb_np = normalize_embeddings(embeddings.cpu().numpy())
            for idx, vec in zip(valid_indices, emb_np):
                results[idx] = vec.tolist()
        except Exception as e:
            print(f"[embed_images] Failed to embed images: {e}")

    return jsonify({"embeddings": results})

# --- Batch Processing ---
def embed_texts(texts, chunk_size=64):
    vectors = []
    for chunk in chunked(texts, chunk_size):
        if not any(chunk):
            vectors.extend([[0.0] * 512] * len(chunk))
            continue
        inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        vectors.extend(normalize_embeddings(embeddings.cpu().numpy()).tolist())
    return vectors

def embed_image_files(image_paths, chunk_size=8):
    vectors = []
    for chunk in chunked(image_paths, chunk_size):
        images = []
        chunk_vecs = []

        for path in chunk:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[image read] Failed for {path}: {e}", flush=True)
                chunk_vecs.append([0.0] * 512)

        if not images:
            vectors.extend(chunk_vecs)
            continue

        try:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            normed = normalize_embeddings(embeddings.cpu().numpy()).tolist()
            vectors.extend(normed)
        except Exception as e:
            print(f"[embed_images] Failed chunk: {e}", flush=True)
            vectors.extend([[0.0] * 512] * len(images))

    return vectors


def process_batch(batch_dir, webhook_url, batch_id, shop):
    try:
        print("Starting batch processing.", flush=True)

        metadata_path = os.path.join(batch_dir, "metadata.json")
        with open(metadata_path) as f:
            meta = json.load(f)

        texts = [f"{m['product_title']} - {m['variant_title']}. Tags: {', '.join(m['tags'])}" for m in meta]
        sizes = [f"Size {m['size']}" if m.get("size") else "" for m in meta]
        colors = [f"Color {m['color']}" if m.get("color") else "" for m in meta]
        image_paths = [os.path.join(batch_dir, "images", m["image_filename"]) if m.get("image_filename") else None for m in meta]

        text_vecs = embed_texts(texts)
        size_vecs = embed_texts(sizes)
        color_vecs = embed_texts(colors)

        image_paths_filtered = [p for p in image_paths if p]
        image_vecs = embed_image_files(image_paths_filtered)

        results = []
        img_index = 0
        for i, m in enumerate(meta):
            img_vec = [0.0] * 512
            body_shape_tags = []
            if image_paths[i]:
                img_vec = image_vecs[img_index]
                img_index += 1
                body_shape_tags = tag_body_shapes_for_image(img_vec, body_shape_embedding_store)
                # log(f"Body shape tag: {body_shape_tags}")


            results.append({
                "variant_id": m["variant_id"],
                "product_id": m["product_id"],
                "shop": m["shop"],
                "embedding": safe_vector(text_vecs[i]),
                "size_embedding": safe_vector(size_vecs[i]),
                "color_embedding": safe_vector(color_vecs[i]),
                "image_embedding": safe_vector(img_vec),
                "body_shapes": body_shape_tags,
                "metadata": m
            })


        response = requests.post(webhook_url, json={"batch_id": batch_id, "results": results, "shop": shop})
        print(f"[webhook] Sent results for batch {batch_id}: {response.status_code}", flush=True)
    except Exception as e:
        print(f"[process_batch] Uncaught exception: {e}", flush=True)


@app.route("/start_batch", methods=["POST"])
def start_batch():
    batch_id = request.form.get("batch_id")
    webhook_url = request.form.get("webhook_url")
    shop = request.form.get("shop")
    if not batch_id or not webhook_url:
        return jsonify({"error": "Missing batch_id or webhook_url"}), 400

    batch_file = request.files.get("batch")
    if not batch_file:
        return jsonify({"error": "Missing batch file"}), 400

    work_dir = f"/tmp/spot_batch_{batch_id}"
    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, "batch.zip")
    batch_file.save(zip_path)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(work_dir)

    Thread(target=process_batch, args=(work_dir, webhook_url, batch_id, shop)).start()
    return jsonify({"status": "accepted", "batch_id": batch_id}), 202

from io import BytesIO
import torch.nn.functional as F

@app.route("/similarity_image_text", methods=["POST"])
def similarity_image_text():
    if "image" not in request.files or "query" not in request.form:
        return jsonify({"error": "Missing 'image' file or 'query' field"}), 400

    try:
        image_file = request.files["image"]
        query = request.form["query"]

        # Load and preprocess image
        image = Image.open(BytesIO(image_file.read())).convert("RGB")
        inputs = processor(text=[query], images=[image], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_emb = model.get_text_features(**{k: inputs[k] for k in inputs if k.startswith("input_ids") or k.startswith("attention_mask")})
            image_emb = model.get_image_features(**{k: inputs[k] for k in inputs if k.startswith("pixel_values")})

        # Normalize
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        image_emb = F.normalize(image_emb, p=2, dim=-1)

        similarity = torch.cosine_similarity(text_emb, image_emb).item()

        return jsonify({
            "query": query,
            "cosine_similarity": round(similarity, 4)
        })

    except Exception as e:
        return jsonify({"error": f"Failed to compute similarity: {str(e)}"}), 500

# --- Health check ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

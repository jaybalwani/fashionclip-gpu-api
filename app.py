from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import requests

app = Flask(__name__)

# Load model & processor
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Auto device detection
device = torch.device("cpu")
model.to(device)

def normalize_embeddings(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)

@app.route("/embed_text", methods=["POST"])
def embed_text():
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Missing or invalid 'texts' list"}), 400

    try:
        # Add explicit max_length=77 to prevent tokenizer overflow
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        emb_np = embeddings.cpu().numpy()
        emb_np = normalize_embeddings(emb_np)
        return jsonify({"embeddings": emb_np.tolist()})

    except Exception as e:
        return jsonify({"error": f"Failed to process text: {str(e)}"}), 500

@app.route("/embed_images", methods=["POST"])
def embed_images():
    data = request.get_json()
    image_urls = data.get("image_urls", [])
    if not image_urls or not isinstance(image_urls, list):
        return jsonify({"error": "Missing or invalid 'image_urls' list"}), 400

    images = []
    for url in image_urls:
        try:
            img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            images.append(img)
        except Exception as e:
            return jsonify({"error": f"Failed to load image {url}: {str(e)}"}), 400

    try:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        emb_np = embeddings.cpu().numpy()
        emb_np = normalize_embeddings(emb_np)
        return jsonify({"embeddings": emb_np.tolist()})
    except Exception as e:
        return jsonify({"error": f"Failed to hiprocess images: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

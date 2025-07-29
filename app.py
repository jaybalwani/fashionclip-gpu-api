from flask import Flask, request, jsonify
import torch
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
from batch_utilities.tasks import sync_products_task
from batch_utilities.product_sync import safe_vector, embed_texts, processor, model, device, normalize_embeddings

app = Flask(__name__)

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

#-------------- MAIN BATCH PROCESSING ENDPOINT -------------------#

@app.route("/start_batch", methods=["POST"])
def start_batch():
    try:

        data = request.get_json()
        shop = data.get("shop", None)
        token = data.get("token", None)

        sync_products_task.delay(shop, token)

        return jsonify({"status": "accepted"}), 202
    except Exception as e:
        print(f"Error in starting batch: {e}")


#------------------ TEST ENDPOINTS --------------------------------------------------------#

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
    
@app.route("/similarity_text", methods=["POST"])
def similarity_text():
    query1 = request.form.get("query1")
    query2 = request.form.get("query2")

    if not query1 or not query2:
        return jsonify({"error": "Both 'query1' and 'query2' fields are required"}), 400

    try:
        embeddings = embed_texts([query1, query2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return jsonify({
            "query1": query1,
            "query2": query2,
            "cosine_similarity": round(float(sim), 4)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to compute similarity: {str(e)}"}), 500


# --- Health check ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

from PIL import Image
import numpy as np
import webcolors
import requests
import os
import uuid
import json
import shutil
from io import BytesIO
from dotenv import load_dotenv
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
from db.db_connect import connection
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk



load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 4))
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://127.0.0.1:9200")

def safe_vector(vector, dim=512):
    if not vector or not isinstance(vector, list) or len(vector) != dim:
        return [1.0 / (dim ** 0.5)] * dim
    norm = np.linalg.norm(vector)
    if norm == 0:
        return [1.0 / (dim ** 0.5)] * dim
    return vector


def extract_option(options, key):
    for option in options:
        if option["name"].lower() == key.lower():
            return option["value"]
    return None


def get_products(shop, access_token):
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }

    products = []
    cursor = None
    has_next_page = True

    while has_next_page:
        query = """
        query ($cursor: String) {
          products(first: 250, after: $cursor) {
            pageInfo {
              hasNextPage
              endCursor
            }
            edges {
              cursor
              node {
                id
                handle
                title
                productType
                description
                tags
                onlineStoreUrl
                images(first: 1) {
                  edges {
                    node {
                      originalSrc
                    }
                  }
                }
                variants(first: 100) {
                  edges {
                    node {
                      id
                      title
                      sku
                      price
                      image {
                        originalSrc
                      }
                      selectedOptions {
                        name
                        value
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        variables = {"cursor": cursor} if cursor else {}

        response = requests.post(
            f"https://{shop}/admin/api/2023-10/graphql.json",
            headers=headers,
            json={"query": query, "variables": variables}
        )

        try:
            data = response.json()
        except Exception as e:
            print("Failed to parse JSON:", e)
            return products

        if "errors" in data:
            print("Shopify returned errors:", data["errors"])
            return products

        edges = data["data"]["products"]["edges"]
        for edge in edges:
            products.append(edge["node"])

        page_info = data["data"]["products"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        cursor = page_info["endCursor"]

    return products


def download_image(url, path):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(path)
    except Exception as e:
        print(f"[img] Failed to download {url}: {e}")


def process_products(shop, access_token):
    try:
        print("Fetching products from Shopify...", flush=True)
        products = get_products(shop, access_token)
        product_count = len(products)
        print(f"Fetched {product_count} products.", flush=True)

        batch_id = str(uuid.uuid4())
        base_dir = f"/tmp/batch_{batch_id}"
        os.makedirs(f"{base_dir}/images", exist_ok=True)

        items = []
        print("Processing products and downloading images..", flush=True)
        for product in products:  
            tags = product.get("tags", [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

            fallback_img = None
            if product.get("images", {}).get("edges"):
                fallback_img = product["images"]["edges"][0]["node"]["originalSrc"]

            for edge in product["variants"]["edges"]:
                variant = edge["node"]
                opts = variant.get("selectedOptions", [])
                size = extract_option(opts, "Size")
                color = extract_option(opts, "Color")
                img_url = (variant.get("image") or {}).get("originalSrc") or fallback_img
                img_filename = f"{variant['id'].split('/')[-1]}.jpg" if img_url else None
                img_path = f"{base_dir}/images/{img_filename}" if img_url else None
                if img_url:
                    download_image(img_url, img_path)

                items.append({
                    "shop": shop,
                    "variant_id": variant["id"],
                    "product_id": product["id"],
                    "product_title": product["title"],
                    "variant_title": variant["title"],
                    "description": product.get("description", ""),
                    "tags": tags,
                    "sku": variant["sku"],
                    "price": float(variant.get("price", 0.0)),
                    "size": size,
                    "color": color,
                    "image_url": img_url if img_url else None,
                    "product_url": product.get("onlineStoreUrl") or f"https://{shop}/products/{product.get('handle')}",
                    "image_filename": img_filename,
                    "selected_options": opts,
                    "product_type": product.get("productType", ""),

                })
    
   
        conn, cur = connection(dictFlag=True)
        
        cur.execute("UPDATE shopify_auth SET total_products = %s WHERE shop = %s;", (product_count,shop,))

        return batch_id, base_dir, items
        
    except Exception as e:
        print(f"[{shop}] Failed to process products: {e}")
        shutil.rmtree(base_dir, ignore_errors=True)

    finally:
        conn.commit()
        cur.close()
        conn.close()


#------------------------------- PROCESS BATCH BEGINS HERE --------------------------------------------------#

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import requests
import os
import json
import uuid
from more_itertools import chunked
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import sys
import re

# Load model & processor
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Auto device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load precomputed body shape clothing embeddings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PKL_PATH = os.path.join(BASE_DIR, "body_shape_embeddings.pkl")

# Now open the file safely
with open(PKL_PATH, "rb") as f:
    body_shape_embedding_store = pickle.load(f)

# --- Utility ---
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

def clean_product_type(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

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

def embed_image_files(image_paths, chunk_size=CHUNK_SIZE):
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



def process_batch(batch_dir, batch_id, shop, items):
    try:
        print("Starting batch processing.", flush=True)
        meta = items


        for i, m in enumerate(meta):
            if not isinstance(m.get("tags"), list):
                print(f"[WARN] meta[{i}] has non-list 'tags': {m.get('tags')}", flush=True)
            if any(tag is None for tag in m.get("tags", [])):
                print(f"[WARN] meta[{i}] has None in 'tags': {m.get('tags')}", flush=True)

        texts =[]
        for i, m in enumerate(meta):
            try:
                joined_tags = ", ".join([t for t in m.get("tags", []) if t is not None])
                text = f"{m['product_title']} - {m['variant_title']}. Tags: {joined_tags}"
                texts.append(text)
            except Exception as e:
                print(f"[ERROR] Failed to create text for meta[{i}]: {m}", flush=True)
                raise
        product_types = [clean_product_type(m.get("product_type", "")) for m in meta]
        sizes = [f"Size {m['size']}" if m.get("size") else "" for m in meta]
        colors = [f"Color {m['color']}" if m.get("color") else "" for m in meta]
        image_paths = [os.path.join(batch_dir, "images", m["image_filename"]) if m.get("image_filename") else None for m in meta]
        
        print("Now embedding texts", flush=True)
        text_vecs = embed_texts(texts)

        print("Now embedding product_types", flush=True)
        product_type_vecs = embed_texts(product_types)

        print("Now embedding sizes", flush=True)
        size_vecs = embed_texts(sizes)

        print("Now embedding colors", flush=True)
        color_vecs = embed_texts(colors)

        image_paths_filtered = [p for p in image_paths if p]

        print("Now embedding images", flush=True)
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
                "product_type_embedding": safe_vector(product_type_vecs[i]),
                "size_embedding": safe_vector(size_vecs[i]),
                "color_embedding": safe_vector(color_vecs[i]),
                "image_embedding": safe_vector(img_vec),
                "body_shapes": body_shape_tags,
                "metadata": m
            })

        conn, cur = connection(dictFlag=True)
        print("Now moving onto ES stuff", flush=True)
        try:
            es = Elasticsearch(
                ELASTIC_URL,
                basic_auth=("elastic", "NBLnx6PCljrZugFTqHNd"),
                verify_certs=False
            )
            actions=[]
            for idx, item in enumerate(results):
                doc_id = quote(f"{item['shop']}__{item['variant_id']}", safe="")
                single_meta = item["metadata"]

                if not isinstance(single_meta.get("tags"), list):
                    print(f"[WARN] result[{idx}] has non-list 'tags': {single_meta.get('tags')}", flush=True)
                if any(tag is None for tag in single_meta.get("tags", [])):
                    print(f"[WARN] result[{idx}] has None in 'tags': {single_meta.get('tags')}", flush=True)


                doc = {
                    "_index": "products",
                    "_id": doc_id,
                    "_source": {
                        "shop": item["shop"],
                        "product_id": item["product_id"],
                        "variant_id": item["variant_id"],
                        "title": single_meta["product_title"],
                        "variant_title": single_meta["variant_title"],
                        "tags": [str(t) for t in single_meta.get("tags", []) if t],
                        "product_type": single_meta["product_type"] or [],
                        "description": single_meta["description"] or [],
                        "sku": single_meta["sku"] or [],
                        "product_url": single_meta["product_url"] or [],
                        "selected_options": single_meta["selected_options"],
                        "price": single_meta["price"],
                        "image_url": single_meta["image_url"] or [],
                        "embedding": safe_vector(item["embedding"]),
                        "product_type_embedding": safe_vector(item["product_type_embedding"]),
                        "size_embedding": safe_vector(item["size_embedding"]),
                        "color_embedding": safe_vector(item["color_embedding"]),
                        "image_embedding": safe_vector(item["image_embedding"]),
                        "body_shapes": item.get("body_shapes", [])
                    }
                }
                actions.append(doc)

            success, failed = bulk(es, actions, raise_on_error=False)
            print(f"[bulk] success={success}, failed={len(failed)}", flush=True)

            cur.execute("UPDATE shopify_auth SET is_synced = 1, is_sync_in_progress = 0 WHERE shop = %s;", (shop,))
            conn.commit()
            print(f"[SUCCESS] Batch processing complete!", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to build doc for result: {e}", flush=True)
            print(f"Metadata that failed: {single_meta}", flush=True)
        finally:
            cur.close()
            conn.close()

    except Exception as e:
        print(f"[process_batch] Uncaught exception: {e}", flush=True)
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)
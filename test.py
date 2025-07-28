import pickle

with open("body_shape_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

for key in list(data.keys())[:3]:
    print(f"{key}:")
    nested = data[key]
    for subkey, value in nested.items():
        print(f"  {subkey}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")

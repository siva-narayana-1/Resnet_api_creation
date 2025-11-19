import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

UPLOAD_DIR = "uploads"
MODEL_DIR = "tflite_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GCS_BUCKET = "https://storage.googleapis.com/food-classification-models-bucket"


def normalize(text):
    return text.lower().strip().replace(" ", "_")


# -----------------------------
# CHECK API STATUS
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "ResNet API is running!", 200


# -----------------------------
# Load Files
# -----------------------------
CLASS_JSON = json.load(open("class.json"))
RESNET_JSON = json.load(open("model_evaluation_results_resnet.json"))
RESNET_JSON = {normalize(k): v for k, v in RESNET_JSON.items()}


RESNET_MODEL_CLASS_INDEX = {
    "resnet_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "resnet_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "resnet_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "resnet_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "resnet_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "resnet_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "resnet_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "resnet_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "resnet_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "resnet_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "resnet_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},
}

RESNET_MODEL_CLASS_INDEX = {
    m.lower(): {normalize(k): v for k, v in d.items()}
    for m, d in RESNET_MODEL_CLASS_INDEX.items()
}


# -----------------------------
# TFLite Loader
# -----------------------------
tflite_cache = {}


def download_resnet_model(model_name):
    url = f"{GCS_BUCKET}/{model_name}.tflite"
    save_path = os.path.join(MODEL_DIR, model_name + ".tflite")

    print("[INFO] Downloading:", url)
    resp = requests.get(url, timeout=45)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

    print("[INFO] Saved:", save_path)
    return save_path


def load_resnet_model(model_name):
    global tflite_cache
    tflite_cache = {}  # keep only 1

    local_path = os.path.join(MODEL_DIR, model_name + ".tflite")

    if not os.path.exists(local_path):
        download_resnet_model(model_name)

    interpreter = tf.lite.Interpreter(model_path=local_path)
    interpreter.allocate_tensors()

    tflite_cache[model_name] = interpreter
    return interpreter


def preprocess(img, size):
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_resnet():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    selected_class = request.form.get("selected_class", "")
    cname = normalize(selected_class)

    fpath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(fpath)

    class_info = RESNET_JSON.get(cname)
    if class_info is None:
        return jsonify({"success": False, "error": "Class not found"})

    model_name = class_info["model_used"].lower()

    interpreter = load_resnet_model(model_name)

    input_info = interpreter.get_input_details()[0]
    _, h, w, _ = input_info["shape"]

    img = Image.open(fpath).convert("RGB")
    x = preprocess(img, w)

    interpreter.set_tensor(input_info["index"], x)
    interpreter.invoke()

    output = interpreter.get_output_details()[0]
    preds = interpreter.get_tensor(output["index"]).squeeze()
    preds = np.nan_to_num(preds)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # decode class
    class_map = RESNET_MODEL_CLASS_INDEX[model_name]
    predicted_label = next(k for k, v in class_map.items() if v == idx)

    os.remove(fpath)

    return jsonify({
        "success": True,
        "model_used": model_name,
        "selected_class": selected_class,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "metrics": class_info
    }), 200


# -----------------------------
# RUN (Railway)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

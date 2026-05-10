import os
import io
import base64
import warnings
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    import joblib
except ImportError:
    joblib = None


app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
ACTIVE_MODEL_FILE = os.path.join(MODEL_DIR, '.active_model')
DEFAULT_MODEL_NAME = 'thai_digits_cnn_model_best.keras'
ALLOWED_EXTENSIONS = {'pkl', 'joblib', 'pt', 'pth', 'h5', 'keras'}

app.config['UPLOAD_FOLDER'] = MODEL_DIR
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

LABELS = ["๒๖", "๒๗", "๒๘", "๒๙", "๓๐"]
LABEL_MAP = {26: "๒๖", 27: "๒๗", 28: "๒๘", 29: "๒๙", 30: "๓๐"}

global_model = None
active_model_info = {}
active_model_name = DEFAULT_MODEL_NAME


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_model_filename(filename):
    if not filename:
        return None
    filename = os.path.basename(filename)
    return filename if allowed_file(filename) else None


def get_model_path(filename):
    filename = normalize_model_filename(filename)
    if not filename:
        return None
    model_path = os.path.abspath(os.path.join(MODEL_DIR, filename))
    if os.path.commonpath([MODEL_DIR, model_path]) != MODEL_DIR:
        return None
    return model_path


def read_saved_active_model():
    try:
        if os.path.exists(ACTIVE_MODEL_FILE):
            with open(ACTIVE_MODEL_FILE, 'r', encoding='utf-8') as fh:
                filename = normalize_model_filename(fh.read().strip())
                model_path = get_model_path(filename)
                if filename and model_path and os.path.exists(model_path):
                    return filename
    except OSError as exc:
        print(f"Failed to read active model file: {exc}")
    return DEFAULT_MODEL_NAME


def save_active_model_name(filename):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(ACTIVE_MODEL_FILE, 'w', encoding='utf-8') as fh:
            fh.write(filename)
    except OSError as exc:
        print(f"Failed to save active model file: {exc}")


active_model_name = read_saved_active_model()


def get_extension(filename):
    return filename.rsplit('.', 1)[1].lower()


def get_model_kind(filename):
    ext = get_extension(filename)
    if ext in {'keras', 'h5'}:
        return 'keras'
    if ext in {'joblib', 'pkl'}:
        return 'sklearn'
    if ext in {'pt', 'pth'}:
        return 'torch'
    return 'unknown'


def load_model_file(filename, model_path=None):
    filename = normalize_model_filename(filename)
    model_path = model_path or get_model_path(filename)
    if not filename or not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")

    kind = get_model_kind(filename)

    if kind == 'keras':
        model = tf.keras.models.load_model(model_path)
        return model, {
            "kind": "keras",
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
        }

    if kind == 'sklearn':
        if joblib is None:
            raise RuntimeError("joblib is not installed")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_object = joblib.load(model_path)

        estimator = model_object.get('model') if isinstance(model_object, dict) else model_object
        if estimator is None or not hasattr(estimator, 'predict'):
            raise ValueError("This file is not a usable prediction model")

        return model_object, {
            "kind": "sklearn",
            "label": model_object.get('label') if isinstance(model_object, dict) else type(estimator).__name__,
            "classes": [int(c) if isinstance(c, np.integer) else c for c in getattr(estimator, 'classes_', [])],
            "n_features": getattr(estimator, 'n_features_in_', None),
        }

    if kind == 'torch':
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed. Install torch to use .pt/.pth models.") from exc

        try:
            model = torch.jit.load(model_path, map_location='cpu')
        except Exception:
            model = torch.load(model_path, map_location='cpu')

        if not callable(model):
            raise ValueError("This .pt/.pth file is not a callable model. Use TorchScript or a full nn.Module.")
        if hasattr(model, 'eval'):
            model.eval()
        return model, {"kind": "torch"}

    raise ValueError("Unsupported model type")


def load_active_model():
    global global_model, active_model_info
    try:
        global_model, active_model_info = load_model_file(active_model_name)
        print(f"Successfully loaded model: {active_model_name}")
    except Exception as exc:
        print(f"Error loading model {active_model_name}: {exc}")
        global_model = None
        active_model_info = {}


def decode_request_image(data_url):
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    image_bytes = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_bytes))


def prepare_base_image(img, mode='L'):
    img = img.convert(mode)
    return ImageOps.invert(img)


def prepare_keras_input(img, input_shape):
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported Keras input shape: {input_shape}")

    _, height, width, channels = input_shape
    height = height or 64
    width = width or 64
    channels = channels or 1

    if channels == 1:
        img = prepare_base_image(img, 'L').resize((width, height), Image.LANCZOS)
        return np.array(img, dtype=np.float32).reshape(1, height, width, 1)

    if channels == 3:
        img = prepare_base_image(img, 'RGB').resize((width, height), Image.LANCZOS)
        return np.array(img, dtype=np.float32).reshape(1, height, width, 3)

    raise ValueError(f"Unsupported channel count: {channels}")


def infer_flat_image_size(n_features):
    if n_features == 2048:
        return 'L', (64, 32)
    if n_features == 12288:
        return 'RGB', (64, 64)

    root = int(np.sqrt(n_features))
    if root * root == n_features:
        return 'L', (root, root)

    if n_features % 3 == 0:
        root = int(np.sqrt(n_features // 3))
        if root * root * 3 == n_features:
            return 'RGB', (root, root)

    raise ValueError(f"Cannot infer image size for {n_features} input features")


def apply_sklearn_preprocessors(features, preprocessors):
    for key in ('scaler', 'pca', 'mm'):
        transformer = preprocessors.get(key)
        if transformer is not None:
            features = transformer.transform(features)
    return features


def prepare_sklearn_input(img, model_object):
    estimator = model_object.get('model') if isinstance(model_object, dict) else model_object
    preprocessors = model_object.get('preprocessors', {}) if isinstance(model_object, dict) else {}
    img_size = model_object.get('img_size') if isinstance(model_object, dict) else None

    if img_size:
        mode, size = 'L', tuple(img_size)
    else:
        n_features = getattr(estimator, 'n_features_in_', None)
        if n_features is None:
            raise ValueError("The sklearn model does not expose n_features_in_")
        mode, size = infer_flat_image_size(int(n_features))

    img = prepare_base_image(img, mode).resize(size, Image.LANCZOS)
    features = np.array(img, dtype=np.float32).reshape(1, -1)
    return apply_sklearn_preprocessors(features, preprocessors)


def predict_keras(model, info, img):
    model_input = prepare_keras_input(img, info.get('input_shape', model.input_shape))
    predictions = np.asarray(model.predict(model_input, verbose=0))
    scores = predictions[0]
    class_idx = int(np.argmax(scores))
    return LABELS[class_idx], float(scores[class_idx])


def predict_sklearn(model_object, img):
    estimator = model_object.get('model') if isinstance(model_object, dict) else model_object
    model_input = prepare_sklearn_input(img, model_object)

    if hasattr(estimator, 'predict_proba'):
        probabilities = estimator.predict_proba(model_input)[0]
        class_idx = int(np.argmax(probabilities))
        predicted_class = estimator.classes_[class_idx]
        confidence = float(probabilities[class_idx])
    else:
        predicted_class = estimator.predict(model_input)[0]
        confidence = 1.0

    predicted_class = int(predicted_class) if isinstance(predicted_class, np.integer) else predicted_class
    return LABEL_MAP.get(predicted_class, str(predicted_class)), confidence


def predict_torch(model, img):
    import torch
    img = prepare_base_image(img, 'L').resize((64, 64), Image.LANCZOS)
    model_input = np.array(img, dtype=np.float32).reshape(1, 1, 64, 64) / 255.0
    with torch.no_grad():
        output = model(torch.from_numpy(model_input))
        if isinstance(output, (list, tuple)):
            output = output[0]
        scores = output.detach().cpu().numpy()[0]
    class_idx = int(np.argmax(scores))
    return LABELS[class_idx], float(scores[class_idx])


@app.route('/')
def user_page():
    return send_from_directory('../frontend', 'user.html')


@app.route('/admin')
def admin_page():
    return send_from_directory('../frontend', 'admin.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)


@app.route('/api/predict', methods=['POST'])
def predict():
    global global_model, active_model_info
    if global_model is None:
        load_active_model()

    if global_model is None:
        return jsonify({"error": "No model loaded and failed to load default"}), 500

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        img = decode_request_image(data['image'])
        kind = active_model_info.get('kind')

        if kind == 'keras':
            result_label, confidence = predict_keras(global_model, active_model_info, img)
        elif kind == 'sklearn':
            result_label, confidence = predict_sklearn(global_model, img)
        elif kind == 'torch':
            result_label, confidence = predict_torch(global_model, img)
        else:
            return jsonify({"error": "Active model type is not supported"}), 500

        return jsonify({
            "prediction": result_label,
            "confidence": confidence,
            "model": active_model_name,
        })

    except Exception as exc:
        print("Prediction error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    if not os.path.exists(MODEL_DIR):
        return jsonify({"models": [], "active_model": active_model_name})

    files = []
    thai_months = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

    for filename in os.listdir(MODEL_DIR):
        if allowed_file(filename):
            filepath = os.path.join(MODEL_DIR, filename)
            stat = os.stat(filepath)
            size_bytes = stat.st_size
            kind = 'preprocessor' if filename.lower() == 'preprocessors.joblib' else get_model_kind(filename)
            usable = kind != 'preprocessor'

            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{(size_bytes / 1024):.1f} KB"
            else:
                size_str = f"{(size_bytes / (1024 * 1024)):.1f} MB"

            dt = datetime.fromtimestamp(stat.st_mtime)
            date_str = f"{dt.day} {thai_months[dt.month - 1]} {dt.year} · {dt.strftime('%H:%M')}"

            files.append({
                "id": filename,
                "name": filename,
                "type": kind,
                "usable": usable,
                "size": size_str,
                "date": date_str,
                "active": filename == active_model_name,
            })

    files.sort(key=lambda item: (not item["active"], item["name"].lower()))
    return jsonify({"models": files, "active_model": active_model_name})


@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f".uploading_{filename}")
    file.save(temp_path)

    try:
        new_model, new_info = load_model_file(filename, temp_path)
        os.replace(temp_path, filepath)
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Cannot load this model: {exc}"}), 400

    global active_model_name, global_model, active_model_info
    active_model_name = filename
    global_model = new_model
    active_model_info = new_info
    save_active_model_name(filename)

    return jsonify({"success": True, "message": "Model uploaded successfully", "filename": filename})


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400

    filename = normalize_model_filename(data['filename'])
    filepath = get_model_path(filename)

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Model not found"}), 404

    try:
        new_model, new_info = load_model_file(filename, filepath)
    except Exception as exc:
        return jsonify({"error": f"Failed to load the model: {exc}"}), 500

    global active_model_name, global_model, active_model_info
    active_model_name = filename
    global_model = new_model
    active_model_info = new_info
    save_active_model_name(filename)

    return jsonify({"success": True, "message": f"Switched to {filename}"})


@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400

    filename = normalize_model_filename(data['filename'])
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400
    if filename == active_model_name:
        return jsonify({"error": "Cannot delete active model"}), 400

    filepath = get_model_path(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Model not found"}), 404

    try:
        os.remove(filepath)
        return jsonify({"success": True, "message": f"Deleted {filename}"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

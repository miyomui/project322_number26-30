import os
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime


# Initialize Flask App
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

# Configuration
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__name__), 'model'))
ALLOWED_EXTENSIONS = {'pkl', 'joblib', 'pt', 'h5', 'keras'}
app.config['UPLOAD_FOLDER'] = MODEL_DIR
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50 MB max

# Global state for active model
global_model = None
active_model_name = 'thai_digits_cnn_model_best.keras'
LABELS = ["๒๖", "๒๗", "๒๘", "๒๙", "๓๐"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_active_model():
    global global_model
    model_path = os.path.join(MODEL_DIR, active_model_name)
    try:
        if os.path.exists(model_path):
            global_model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model: {active_model_name}")
        else:
            print(f"Model not found at: {model_path}")
            global_model = None
    except Exception as e:
        print(f"Error loading model {active_model_name}: {e}")
        global_model = None

# Initial model load
load_active_model()

# --- Page Routes ---

@app.route('/')
def user_page():
    return send_from_directory('../frontend', 'user.html')

@app.route('/admin')
def admin_page():
    return send_from_directory('../frontend', 'admin.html')

# Serve static files like CSS/JS if any (not strictly needed since we use CDN and inline mostly, but good practice)
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

# --- API Routes ---

@app.route('/api/predict', methods=['POST'])
def predict():
    if global_model is None:
        return jsonify({"error": "No model loaded"}), 500

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # data['image'] looks like "data:image/png;base64,iVBORw0KGgo..."
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Load image with Pillow
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Invert colors: canvas is usually white background with black drawing
        # Neural nets typically prefer black background with white drawing (like MNIST)
        # We can check the model's expected background, but MNIST is inverted.
        # Let's invert to be safe (black background, white digit)
        import PIL.ImageOps
        img = PIL.ImageOps.invert(img)
        
        # Resize to 64x64
        img = img.resize((64, 64), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize (assuming the model expects 0-255 or 0-1. The summary shows a Rescaling layer!)
        # The summary says: `rescaling (Rescaling) | (None, 64, 64, 1)`
        # This implies the model expects raw pixel values (0-255) and scales them internally.
        
        # Reshape to (1, 64, 64, 1)
        img_array = img_array.reshape(1, 64, 64, 1)
        
        # Predict
        predictions = global_model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        result_label = LABELS[class_idx]
        
        return jsonify({
            "prediction": result_label,
            "confidence": confidence
        })
        
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    if not os.path.exists(MODEL_DIR):
        return jsonify({"models": []})
        
    files = []
    for filename in os.listdir(MODEL_DIR):
        if allowed_file(filename):
            filepath = os.path.join(MODEL_DIR, filename)
            stat = os.stat(filepath)
            
            # Format size
            size_bytes = stat.st_size
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{(size_bytes / 1024):.1f} KB"
            else:
                size_str = f"{(size_bytes / (1024 * 1024)):.1f} MB"
                
            # Format date (Thai format rough approximation)
            dt = datetime.fromtimestamp(stat.st_mtime)
            thai_months = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
            date_str = f"{dt.day} {thai_months[dt.month - 1]} {dt.year} · {dt.strftime('%H:%M')}"
            
            files.append({
                "id": filename,
                "name": filename,
                "size": size_str,
                "date": date_str,
                "active": (filename == active_model_name)
            })
            
    # Sort files: active model first, then alphabetically
    files.sort(key=lambda x: (not x["active"], x["name"]))
    return jsonify({"models": files, "active_model": active_model_name})

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Set as active model and reload
        global active_model_name
        active_model_name = filename
        load_active_model()
        
        return jsonify({"success": True, "message": "Model uploaded successfully", "filename": filename})
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400
        
    filename = data['filename']
    filepath = os.path.join(MODEL_DIR, filename)
    
    if os.path.exists(filepath) and allowed_file(filename):
        global active_model_name
        active_model_name = filename
        load_active_model()
        
        if global_model is not None:
            return jsonify({"success": True, "message": f"Switched to {filename}"})
        else:
            return jsonify({"error": "Failed to load the model"}), 500
    
    return jsonify({"error": "Model not found"}), 404

@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400
        
    filename = data['filename']
    if filename == active_model_name:
        return jsonify({"error": "Cannot delete active model"}), 400
        
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({"success": True, "message": f"Deleted {filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            


if __name__ == '__main__':
    app.run(debug=True, port=5000)

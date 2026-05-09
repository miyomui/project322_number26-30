from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import base64
import time

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

labels = ['26', '27', '28', '29', '30']
for label in labels:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

def get_image_counts():
    counts = {}
    for label in labels:
        folder_path = os.path.join(DATASET_DIR, label)
        counts[label] = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
    return counts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stats', methods=['GET'])
def get_stats():
    return jsonify(get_image_counts())

@app.route('/save_image', methods=['POST'])
def save_image():
    try:
        data = request.json
        label = data['label']
        image_data = data['image'].split(',')[1] 
        
        filename = f"{label}_{int(time.time()*1000)}.png"
        filepath = os.path.join(DATASET_DIR, label, filename)
        
        with open(filepath, "wb") as fh:
            fh.write(base64.b64decode(image_data))
            
        return jsonify({"status": "success", "counts": get_image_counts()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 🌟 API ใหม่: ส่งไฟล์รูปไปให้หน้าเว็บแสดงผล
@app.route('/image/<label>/<filename>')
def serve_image(label, filename):
    return send_from_directory(os.path.join(DATASET_DIR, label), filename)

# 🌟 API ใหม่: ดึงรายชื่อรูปล่าสุด 10 รูป
@app.route('/get_recent', methods=['GET'])
def get_recent():
    recent_images = []
    for label in labels:
        folder_path = os.path.join(DATASET_DIR, label)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    filepath = os.path.join(folder_path, filename)
                    recent_images.append({
                        'label': label,
                        'filename': filename,
                        'time': os.path.getmtime(filepath),
                        'url': f'/image/{label}/{filename}'
                    })
    # เรียงลำดับจากใหม่ไปเก่า และส่งไปแค่ 10 รูป
    recent_images.sort(key=lambda x: x['time'], reverse=True)
    return jsonify(recent_images[:30])

# 🌟 API ใหม่: คำสั่งลบไฟล์
@app.route('/delete_image', methods=['POST'])
def delete_image():
    try:
        data = request.json
        filepath = os.path.join(DATASET_DIR, data['label'], data['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"status": "success", "counts": get_image_counts()})
        return jsonify({"status": "error", "message": "ไม่พบไฟล์"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
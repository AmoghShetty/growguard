from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv
import base64
import os
import time

load_dotenv()

# GrowGuard — AI Crop Disease Detection
app = Flask(__name__)

from flask_talisman import Talisman
Talisman(app, content_security_policy=False, force_https=False)

# ---- SECURITY SETTINGS ----
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}

ALLOWED_CROPS = {
    'wheat', 'rice', 'date palm', 'tomato',
    'potato', 'corn', 'cotton', 'soybean'
}
ALLOWED_REGIONS = {
    'Middle East', 'South Asia', 'Sub-Saharan Africa',
    'North Africa', 'Southeast Asia', 'Latin America'
}

# ---- RATE LIMITING (simple in-memory) ----
request_counts = {}
RATE_LIMIT = 10        # max requests
RATE_WINDOW = 60 * 60  # per hour (in seconds)

def is_rate_limited(ip):
    now = time.time()
    if ip not in request_counts:
        request_counts[ip] = []
    # Remove requests older than the window
    request_counts[ip] = [t for t in request_counts[ip] if now - t < RATE_WINDOW]
    if len(request_counts[ip]) >= RATE_LIMIT:
        return True
    request_counts[ip].append(now)
    return False

def allowed_file(filename, mime_type):
    has_valid_ext = '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    has_valid_mime = mime_type in ALLOWED_MIME_TYPES
    return has_valid_ext and has_valid_mime

# ---- ROUTES ----
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/analyze", methods=["POST"])
def analyze():
    # Rate limiting
    ip = request.remote_addr
    if is_rate_limited(ip):
        return jsonify({
            "error": "Too many requests. Please wait an hour before trying again."
        }), 429

    # Validate image exists
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    # Validate file type
    if not allowed_file(image_file.filename, image_file.content_type):
        return jsonify({
            "error": "Invalid file type. Please upload a JPG, PNG or WebP image."
        }), 400

    # Validate crop and region against whitelist
    crop_type = request.form.get("crop_type", "").strip()
    region = request.form.get("region", "").strip()

    if crop_type not in ALLOWED_CROPS:
        return jsonify({"error": "Invalid crop type selected."}), 400

    if region not in ALLOWED_REGIONS:
        return jsonify({"error": "Invalid region selected."}), 400

    # Read image
    image_data = image_file.read()
    image_type = image_file.content_type

    prompt = f"""You are an expert agricultural plant pathologist.
A farmer in {region} has uploaded a photo of their {crop_type} plant.

Analyze the image and respond in exactly this format:

DISEASE: [name of disease or 'Healthy' if no disease]
SEVERITY: [Mild / Moderate / Severe / None]
CONFIDENCE: [percentage like 85%]
SYMPTOMS: [one sentence describing what you see]
TREATMENT: [2-3 specific actionable steps]
PREVENTION: [1-2 prevention tips for the future]

Be specific, practical and helpful for a real farmer."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_data, mime_type=image_type),
                prompt
            ]
        )
        return jsonify({"result": response.text})

    except Exception as e:
        # Never expose raw errors to the user
        print(f"AI error: {str(e)}")
        return jsonify({
            "error": "Analysis failed. Please try again with a clearer photo."
        }), 500

# ---- ERROR HANDLERS ----
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404

if __name__ == "__main__":
    # debug=False for production
    app.run(debug=False)
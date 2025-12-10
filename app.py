import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# --- Configuration ---
# Set the path where you saved your model on the server
MODEL_PATH = 'generator_model_wgangp_final.h5'
OUTPUT_DIR = 'static/generated_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Global Model Loading ---
try:
    # Load the generator model once when the app starts
    # Note: Use custom_objects if your model uses custom layers/functions
    GENERATOR_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False) 
    print("✅ Generator Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    GENERATOR_MODEL = None
    
# --- Preprocessing/Postprocessing Functions ---

def preprocess_image(image_data):
    """Resizes and normalizes the input image."""
    image = Image.open(BytesIO(image_data))
    image = image.resize((32, 32))
    img_array = np.array(image).astype('float32')
    # Normalize to -1 to 1 (same as training data)
    normalized_array = (img_array / 127.5) - 1.0
    # Add batch dimension
    return np.expand_dims(normalized_array, axis=0)

def postprocess_image(generated_tensor):
    """Denormalizes and converts the generated tensor to JPEG/PNG data."""
    # Denormalize from -1 to 1 to 0-255
    image_array = (generated_tensor[0] + 1.0) * 127.5
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    # Convert numpy array to PIL Image object
    img = Image.fromarray(image_array)
    
    # Save to buffer
    img_io = BytesIO()
    img.save(img_io, format='JPEG')
    return base64.b64encode(img_io.getvalue()).decode('utf-8')


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    if GENERATOR_MODEL is None:
        return jsonify({'error': 'Model not available.'}), 503
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    
    try:
        # 1. Preprocess
        input_tensor = preprocess_image(image_data)
        
        # 2. Predict (Run GAN)
        generated_tensor = GENERATOR_MODEL.predict(input_tensor)
        
        # 3. Postprocess and Encode
        encoded_image = postprocess_image(generated_tensor)
        
        return jsonify({
            'success': True,
            # Send the base64 encoded image string
            'generated_image': f'data:image/jpeg;base64,{encoded_image}' 
        })
        
    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return jsonify({'error': f'Processing failed: {e}'}), 500


if __name__ == '__main__':
    # Ensure your model file is in the root directory or adjust MODEL_PATH
    app.run(debug=True)
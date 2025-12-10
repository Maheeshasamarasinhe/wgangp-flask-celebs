# 🌟 WGAN-GP CelebSynth: Robust Image-to-Image Face Generator

![Project Screenshot](img/Screenshot%202025-12-10%20214537.png)

## Project Overview
This project implements a highly stable Generative Adversarial Network (GAN) using the Wasserstein GAN with Gradient Penalty (WGAN-GP) architecture to perform Image-to-Image (I2I) Face Synthesis.

The application is built with a Python Flask backend to host the deep learning model and a modern HTML/CSS/JavaScript frontend for user interaction.

## Key Features
*   **Robust Architecture:** Utilizes WGAN-GP to overcome severe instability and Mode Collapse issues often faced by traditional GANs.
*   **Image-to-Image:** Takes a low-resolution image as input and generates a high-quality, synthesized celebrity-like face as output.
*   **Scalable Backend:** Uses Flask to serve the model predictions efficiently.
*   **Technology Stack:** TensorFlow/Keras, Flask, HTML/CSS/JS.

## 🚀 Getting Started
Follow these steps to set up and run the project locally.

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone https://github.com/Maheeshasamarasinhe/wgangp-flask-celebs.git
cd wgangp-flask-celebs
```

### 2. Environment Setup
It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt
```
*(Note: You must create a requirements.txt file containing the following: Flask, tensorflow, numpy, Pillow)*

### 3. Model Acquisition
This application requires the pre-trained WGAN-GP Generator Model.

**Download the Model:** Place your trained model file (`generator_model_wgangp_final.h5`) into the root directory of this project.

### 4. Running the Application

```bash
# Run the Flask application
python app.py
```

The application should start, and you will see a message similar to: `Running on http://127.0.0.1:5000/`

Open your web browser and navigate to the address provided.

## ⚙️ Architecture and Training Details

| Component | Role in the GAN | Technology Used |
| :--- | :--- | :--- |
| **Generator** | Takes input image, outputs synthesized face. | Encoder-Decoder CNN (U-Net style) with tanh activation. |
| **Critic** | Estimates the Wasserstein distance (score) between real and fake images. | CNN with Linear final activation (No Sigmoid). |
| **Loss Function** | Stabilizes training and prevents collapse. | WGAN-GP Loss (Wasserstein Distance + Gradient Penalty, $\lambda=10$). |
| **Optimizer** | Manages learning rates. | Adam or RMSprop (with adjusted parameters). |
| **Training Ratio** | Prioritizes Critic training over Generator. | D:G Ratio of 4:1 or 5:1. |

## 📚 Code Structure

*   **`app.py`**: The main Flask backend. Handles model loading, preprocessing, the `/generate` API endpoint, and rendering the frontend.
*   **`generator_model_wgangp_final.h5`**: The pre-trained Keras/TensorFlow Generator Model.
*   **`templates/index.html`**: The frontend (HTML/JS) file. Manages image upload, preview, and AJAX calls to the backend.
*   **`requirements.txt`**: Python dependency list.

## 🤝 Contribution
Contributions are welcome! If you find any issues or have suggestions, please open an issue or submit a pull request.

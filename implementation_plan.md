# Aircraft Damage Classification and Captioning System

This implementation plan outlines the steps to build a complete web application for detecting and captioning aircraft damage (specifically Cracks and Dents) using a transfer learning model (MobileNetV2) with a Flask backend and a premium glassmorphic frontend.

## Proposed System Features

1. **Synthetic Dataset Generation (`generate_dummy_data.py`)**: Automatically creates synthetic images of cracks (jagged lines) and dents (shaded ellipses) to allow immediate training.
2. **Transfer Learning Pipeline (`train.py`)**: Trains a custom classification head on top of a pretrained MobileNetV2 backbone. Saves the trained model to `model.h5`.
3. **Interactive Grad-CAM Heatmaps**: Computes activation maps to visualize where the model is looking, overlaid on the input image.
4. **Flask Web Application (`app.py`)**: A backend server that handles image upload, classification, Grad-CAM generation, and caption mapping.
5. **Modern Glassmorphic Frontend (`templates/index.html` & `static/css/style.css`)**: A sleek dark-themed dashboard featuring drag-and-drop file upload, live visual previews, confidence meters, and actionable maintenance insights.

---

## Proposed Changes

### 1. Project Configuration & Setup

#### [NEW] [requirements.txt](file:///d:/new%20p/requirements.txt)
Specifies python packages: `tensorflow`, `flask`, `numpy`, `pillow`, `matplotlib`.

### 2. Dataset Generation & Training

#### [NEW] [generate_dummy_data.py](file:///d:/new%20p/generate_dummy_data.py)
A script to generate a small set of synthetic training and test images in the `dataset/train/` and `dataset/test/` folders:
*   **Crack images**: Gray backgrounds with random dark, jagged lines.
*   **Dent images**: Gray backgrounds with shaded, elliptical gradients resembling dents.

#### [NEW] [train.py](file:///d:/new%20p/train.py)
Loads the dataset, instantiates MobileNetV2 with frozen base layers, adds a Classification Head, trains the model, saves `model.h5`, and plots training metrics to `static/metrics.png`.

### 3. Backend Implementation

#### [NEW] [app.py](file:///d:/new%20p/app.py)
The Flask server which:
*   Loads the trained model (`model.h5`). If the model doesn't exist, it uses a fallback/untrained model with a warning to allow testing without waiting for training.
*   Implements Grad-CAM visualization to identify which region of the image triggered the classification.
*   Provides `/predict` endpoints and maps the predicted class to descriptive captions and maintenance suggestions.

### 4. Frontend Design

#### [NEW] [templates/index.html](file:///d:/new%20p/templates/index.html)
The dashboard UI structure containing:
*   A drop-zone for files.
*   Side-by-side visualization showing the uploaded image and the Grad-CAM heatmap.
*   Sleek results panel with damage type, confidence score, dynamic description, and recommendation lists.

#### [NEW] [static/css/style.css](file:///d:/new%20p/static/css/style.css)
The CSS styling using a dark-mode glassmorphic design system:
*   **Fonts**: Outfit or Inter from Google Fonts.
*   **Colors**: Slate dark background, glowing neon blue accents (`#00f0ff`), red for cracks (`#ff4d4d`), orange for dents (`#ffa500`).
*   **Animations**: Pulse effects, slide-ins, and loading animations.

---

## Verification Plan

### Automated/Local Tests
1.  **Environment Setup**: Verify dependencies install correctly.
2.  **Dataset Generation**: Run `generate_dummy_data.py` and inspect generated folders.
3.  **Training**: Run `train.py` to ensure model trains and exports `model.h5` and `static/metrics.png`.
4.  **Web Server**: Run `app.py` and access the dashboard.
5.  **Inspection Flow**: Upload dummy images of cracks and dents and verify classification, captioning, and Grad-CAM output.

### Manual Verification
*   Open the browser interface, perform upload and verify styling alignment and responsiveness.

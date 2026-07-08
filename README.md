# ✈️ Aircraft Surface Fault Detection

A web-based AI inspection tool that detects surface damage on aircraft components using a custom-trained **YOLOv8** model. Upload an image and instantly identify **corrosion**, **cracks**, and **dents** with maintenance recommendations.

---

## 🚀 Features

- 🔍 Real-time detection of corrosion, cracks, and dents
- 📸 Upload aircraft surface images for instant analysis
- 🖼️ Annotated output image with bounding boxes
- 📋 Automated maintenance recommendations per defect type
- ⚡ Powered by YOLOv8 + Flask

---

## 🖥️ Demo

> Run locally and open `http://127.0.0.1:5000` in your browser.

---

## 📁 Project Structure

```
├── app.py                  # Flask web application
├── train.py                # Model training script
├── best.pt                 # Trained YOLOv8 model weights
├── data_local.yaml         # Dataset config (for training only)
├── requirements.txt        # Python dependencies
├── templates/
│   ├── landing.html        # Landing page
│   └── index.html          # Inspection page
└── static/
    ├── css/                # Stylesheets
    └── uploads/            # Uploaded & annotated images (auto-created)
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```
.\venv\Scripts\Activate.ps1 pw


**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python app.py
```

### 6. Open in browser
```
http://127.0.0.1:5000
```

---

## 🤖 Model

The model (`best.pt`) is a custom-trained **YOLOv8** model trained on aircraft surface damage images.

**Detects 3 classes:**
| Class | Description |
|-------|-------------|
| `corrosion` | Surface coating degradation / rust |
| `crack` | Structural stress fractures |
| `dent` | External impact damage |

> ⚠️ If `best.pt` is missing, the app falls back to a generic YOLOv8n model (not trained for aircraft damage).

---

## 🔁 Retraining (Optional)

To retrain the model on your own dataset:

1. Update `data_local.yaml` with your dataset path
2. Run:
```bash
python train.py
```

> This will generate a new `best.pt` in the `runs/` folder.

---

## 📦 Requirements

- Python 3.8+
- flask
- ultralytics
- opencv-python
- numpy
- pillow
- matplotlib

---

## 📄 License

This project is for educational and research purposes.

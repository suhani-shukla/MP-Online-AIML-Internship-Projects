# 🖼️ CIFAR-10 Image Classification

A convolutional neural network (CNN) trained from scratch on the CIFAR-10 dataset, with a full pipeline from data loading to an interactive Streamlit app for live predictions.

## 📋 Overview

This project classifies 32x32 color images into one of 10 categories:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

It covers the complete ML workflow:
- Data loading & exploratory analysis
- Preprocessing (normalization, augmentation, train/val split)
- CNN model design (Conv2D + BatchNorm + Dropout blocks)
- Training with early stopping & learning rate scheduling
- Evaluation (accuracy, classification report, confusion matrix)
- Single-image inference
- A Streamlit web app for interactive predictions

## 📁 Project Structure

```text
Cifar10ImageClassification/
│
├── data/                              # Dataset (downloaded at runtime, git-ignored)
│   └── .gitkeep
│
├── models/                            # Saved model & class names (git-ignored)
│   ├── cnn_model.keras
│   └── class_names.pkl
│
├── notebooks/
│   └── CIFAR10_Image_Classification.ipynb   # End-to-end walkthrough
│
├── screenshots/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── streamlit_app.png
│
├── src/
│   ├── data_loader.py                 # Loads CIFAR-10 dataset and class names
│   ├── preprocess.py                  # Normalization, encoding, augmentation
│   ├── model.py                       # CNN architecture
│   ├── train.py                       # Training loop and callbacks
│   ├── evaluate.py                    # Test evaluation and diagnostic plots
│   ├── predict.py                     # Single-image inference
│   └── app.py                         # Streamlit application
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd Cifar10ImageClassification

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 🚀 Usage

**Train the model** (downloads CIFAR-10 automatically on first run):
```bash
cd src
python train.py
```

**Evaluate on the test set** (generates confusion matrix + training history plots):
```bash
python evaluate.py
```

**Predict a single image:**
```bash
python predict.py --image path/to/image.png
```

**Generate a sample predictions grid:**
```bash
python predict.py --grid
```

**Launch the interactive app:**
```bash
streamlit run app.py
```

## 🧠 Model Architecture

Three convolutional blocks (Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout) with increasing filter depth (32 → 64 → 128), followed by a dense classifier head with dropout regularization. Data augmentation (random flip, rotation, translation, zoom) is applied as model layers, active only during training.

Trained with:
- Adam optimizer
- Categorical cross-entropy loss
- Early stopping on validation loss
- Learning rate reduction on plateau

## 📊 Results

| Metric          | Value    |
|-----------------|----------|
| Test Accuracy   | _fill in after training_ |
| Test Loss       | _fill in after training_ |

### Training History
![Training History](screenshots/training_history.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### Sample Predictions
![Sample Predictions](screenshots/sample_predictions.png)

### Streamlit App
![Streamlit App](screenshots/streamlit_app.png)

## 🔍 Key Observations

_Fill in after evaluation — e.g. which classes are most commonly confused (cats vs dogs, automobiles vs trucks are typical CIFAR-10 confusions), and how augmentation affected generalization._

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model building & training
- **NumPy / scikit-learn** — data handling & metrics
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — interactive web app
- **Pillow** — image preprocessing

## 🔮 Future Improvements

- Transfer learning with a pretrained backbone (ResNet, EfficientNet)
- Hyperparameter tuning (learning rate, batch size, architecture depth)
- Test-time augmentation
- Deploy the Streamlit app (e.g. Streamlit Community Cloud, Hugging Face Spaces)

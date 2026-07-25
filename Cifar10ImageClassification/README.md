# CIFAR-10 Image Classification

A CNN-based image classifier trained on the CIFAR-10 dataset, with an interactive Streamlit app for live predictions.

## Project Structure

Cifar10ImageClassification/
├── data/ # Dataset (downloaded at runtime, git-ignored)
├── models/ # Saved model + class names (git-ignored)
├── notebooks/ # End-to-end exploratory notebook
├── screenshots/ # Training curves, confusion matrix, app screenshot
├── src/ # Source code (data loading, training, evaluation, app)
├── requirements.txt
└── README.md

## Setup

```bash
git clone <your-repo-url>
cd Cifar10ImageClassification
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Train the model
python src/train.py

# Evaluate on test set
python src/evaluate.py

# Run a single prediction
python src/predict.py --image path/to/image.png

# Launch the Streamlit app
streamlit run src/app.py
```

## Results

_(To be filled in after training — accuracy, loss curves, confusion matrix)_

## License

MIT
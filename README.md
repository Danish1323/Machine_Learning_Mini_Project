# Handwritten Digit Recognition

Mini project for Machine Learning Fundamentals (Group 7).

## What it does
- Trains KNN, Logistic Regression, and MLP Classifier on `load_digits` (8x8) and `MNIST` (28x28).
- Applies PCA (95% variance) and compares accuracy and training time.
- Uses a Confusion Matrix to find the most confused digit pairs.
- Deploys a Streamlit app (`app.py`) to predict digits with confidence scores using either model.

## How to run
1. Run all cells in `digit_recognition.ipynb` using the `ml_mini` kernel to train and save the models.
2. Start the Streamlit app:
   ```bash
   uv run streamlit run app.py
   ```

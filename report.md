# Handwritten Digit Recognition Using Machine Learning and PCA

**Subject:** Machine Learning Fundamentals — Mini Project  
**Group:** Group 7  
**Submitted By:** Omkar Narvekar, Sarthak Mhatre, Jaikishan Suthar, Danish Shaikh  

---

## 1. Abstract
Handwritten digit recognition is a classic multi-class classification problem in machine learning. In this project, we build and compare multiple classification models—K-Nearest Neighbors (KNN), Logistic Regression, and Multi-Layer Perceptron (MLP)—on handwritten digit datasets (`load_digits` 8×8 and `MNIST` 28×28). We also study the effect of dimensionality reduction using Principal Component Analysis (PCA) on model accuracy and training time, analyze misclassifications using a Confusion Matrix, and deploy the trained models in an interactive Streamlit web application.

---

## 2. Introduction and Motivation
Recognizing handwritten digits automatically is used in postal mail sorting, bank check processing, and digitizing forms. Because handwriting varies widely across people, rule-based systems fail, making machine learning ideal for learning patterns directly from pixel intensities.

---

## 3. Problem Statement and Objectives
**Problem Statement:** Build a classification model that recognizes handwritten digits (0–9) from pixel values and study the effect of dimensionality reduction (PCA) and a neural network (MLP).

**Objectives:**
- Normalize pixel data and visualize sample digit images with their labels.
- Implement and compare KNN, Logistic Regression, and MLP Classifier.
- Apply PCA (retaining 95% variance) and compare accuracy and training time with and without PCA.
- Use a Confusion Matrix to identify the most frequently confused digits.
- Deploy a Streamlit interface that predicts a selected or uploaded digit image with confidence.

---

## 4. Literature Survey
Traditional approaches such as KNN and Logistic Regression perform well on low-resolution digit datasets like `load_digits` (8×8), while neural networks (MLP) capture non-linear decision boundaries more effectively on higher-resolution datasets like MNIST (28×28). Dimensionality reduction via PCA is widely used to remove correlated pixel features and speed up training.

---

## 5. Existing System / Limitations
- Low-resolution datasets (8×8) lose fine stroke details, making real-world uploaded photos harder to classify.
- Training on raw high-dimensional pixel data (784 features in MNIST) increases computation time.

---

## 6. Proposed System and Workflow
**Workflow:**
`Problem Definition -> Dataset Loading -> Normalization -> EDA & Visualization -> Train/Test Split -> Model Training (KNN, Logistic Regression, MLP) -> PCA Dimensionality Reduction -> Model Comparison -> Confusion Matrix Analysis -> Model Saving -> Streamlit Deployment`

![Project File Structure](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623642473_ae74bf63.png)

---

## 7. Dataset Description
1. **Scikit-learn `load_digits`:** 1,797 grayscale images of size 8×8 (64 features), pixel values 0–16, target classes 0–9.
2. **MNIST (`mnist_784`):** 70,000 grayscale images of size 28×28 (784 features), pixel values 0–255, target classes 0–9 (used as an extension for higher accuracy on uploaded images).

---

## 8. Data Preprocessing
- **Normalization:** Pixel values scaled to `[0, 1]` (`X / 16.0` for `load_digits`, `X / 255.0` for `MNIST`).
- **Train/Test Split:** Stratified split (80/20 for `load_digits`; 60,000/10,000 for `MNIST`) to preserve class balance.
- **Image Upload Preprocessing (Streamlit):** Grayscale conversion, background inversion, contrast stretching, bounding-box cropping, and resizing to 8×8 or 28×28.

---

## 9. Exploratory Data Analysis (EDA)
- Verified dataset shapes and balanced class distribution across digits 0–9.
- Plotted sample 8×8 and 28×28 digit images with their true labels.

![Sample Digits from load_digits](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623679022_ffa80b2e.png)

---

## 10. Model Development
Trained three multi-class classifiers (0–9):
1. **K-Nearest Neighbors (`KNeighborsClassifier`, k=5)**
2. **Logistic Regression (`LogisticRegression`)**
3. **Multi-Layer Perceptron (`MLPClassifier`)**

Each model was trained both **without PCA** and **with PCA** (`n_components=0.95`).

---

## 11. Model Evaluation and Comparison
- Evaluated using **Accuracy**, **Training Time (seconds)**, and **Confusion Matrix**.
- **Effect of PCA (`load_digits`):** Retaining 95% variance reduced features from **64 to 28**, speeding up Logistic Regression (`0.0393s -> 0.0079s`) and MLP (`0.4300s -> 0.3214s`) while keeping accuracy at **0.9806** (MLP) and **0.9556** (Logistic Regression).
- **Confusion Matrix (`load_digits` MLP):** Most confused pair was `8 -> 1` (3 times), with single misclassifications for `0 -> 4`, `1 -> 8`, `6 -> 8`, and `9 -> 8`.

![load_digits Comparison Table](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623725994_61d2d66c.png)

![load_digits Accuracy and Train Time](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623734977_c5cabfb1.png)

![load_digits Confusion Matrix](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623790467_0993f78a.png)

- **Effect of PCA (`MNIST` 28×28):** Retaining 95% variance reduced features from **784 to 154**, cutting MLP training time nearly in half (`4.0711s -> 2.3383s`) while improving accuracy slightly (`0.9791 -> 0.9797`).

![MNIST Comparison Table](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623894796_c068a0eb.png)

![MNIST Confusion Matrix](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795623912794_dceb9d37.jpg)

---

## 12. Streamlit Deployment
Built `app.py` using Streamlit:
- Allows switching between **MNIST (28×28)** and **load_digits (8×8)** models.
- Supports **selecting a sample digit** or **uploading a handwritten digit image**.
- Displays the predicted digit, confidence percentage, and probability bar chart.

![Streamlit Sample Digit Prediction (MNIST)](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795624028286_a67e1134.png)

![Streamlit Uploaded Image on load_digits (8x8) - Misclassified](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795624118473_ba359c08.png)

![Streamlit Uploaded Image on MNIST (28x28) - 99.99% Confidence](/Users/danishshaikh1423/.gemini/antigravity/brain/d3dbeda6-f738-4933-bb3d-8a24a7b2e1b1/.user_uploaded/media_1795624139153_37011b8b.png)

---

## 13. Results and Discussion
- **MLP** achieved strong accuracy on both `load_digits` (~98%) and `MNIST`.
- **MNIST (28×28)** performed significantly better than `load_digits` (8×8) on real-world uploaded photos because 28×28 preserves thin pen strokes.

---

## 14. Conclusion and Future Scope
**Conclusion:** Multi-class classification with MLP and PCA provides high accuracy and fast training for handwritten digit recognition, and Streamlit makes the model easy to test interactively.

**Future Scope:**
- Use Convolutional Neural Networks (CNNs) for even higher accuracy on noisy real-world photos.
- Add live canvas drawing directly inside the Streamlit app.

---

## 15. Individual Contribution Details
- **Danish Shaikh (Lead):** End-to-end ML pipeline architecture, data normalization & stratified splitting, multi-class KNN/Logistic Regression/MLP modeling, PCA dimensionality reduction analysis, MNIST (28×28) dataset extension, Confusion Matrix & error analysis, and model serialization.
- **Omkar Narvekar:** Exploratory Data Analysis (EDA), sample visualization, and dataset verification.
- **Sarthak Mhatre:** Accuracy & training time benchmarking, comparison tables, and visualization plots.
- **Jaikishan Suthar:** Streamlit UI integration (`app.py`), application testing, and report documentation.

---

## 16. References
1. Scikit-learn Documentation: `load_digits`, `PCA`, `KNeighborsClassifier`, `LogisticRegression`, `MLPClassifier`.
2. Yann LeCun et al., *The MNIST Database of Handwritten Digits*.
3. Streamlit Documentation: https://docs.streamlit.io

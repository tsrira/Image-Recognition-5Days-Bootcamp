# Vision AI in 5 Days: From Beginner to Image Recognition Expert

### 📌 Project Overview
This project was developed during a **5-day Image Recognition Bootcamp** focused on building AI-powered image classification systems. Starting from fundamentals, we progressed to advanced deep learning techniques, creating and deploying robust computer vision models.

The project uses **Python, TensorFlow/Keras**, and **Google Colab** to cover:
- Image preprocessing and augmentation
- Convolutional Neural Networks (CNNs)
- Transfer Learning (MobileNetV2)
- Model evaluation and visualization
- Deployment-ready model saving/loading

---

## 📚 Learning Outcomes
By completing this project, I learned and implemented:
- **Image Preprocessing** – Normalization, resizing, dataset visualization
- **Deep Learning Fundamentals** – CNN architecture and training
- **Data Augmentation** – Rotation, flipping, brightness adjustments
- **Model Optimization** – Transfer learning and fine-tuning
- **Evaluation Metrics** – Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve
- **Portfolio Building** – Clear documentation, visualizations, and demo preparation

---

## 📂 Datasets Used
1. **MNIST** – Handwritten digits  
2. **CIFAR-10** – Object images across 10 categories  
3. **Cats vs Dogs** – Binary image classification  

Sources: TensorFlow datasets & Kaggle.

---

## 🛠 Tech Stack
- **Languages**: Python
- **Frameworks/Libraries**: TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn
- **Environment**: Google Colab
- **Pre-trained Model**: MobileNetV2 (ImageNet weights)

---

## 📈 Model Architectures
### 1. **Custom CNN**
- Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output
- Trained from scratch on MNIST and CIFAR-10 datasets

### 2. **Transfer Learning with MobileNetV2**
- Base model with frozen layers
- Global Average Pooling
- Dense layers for binary classification (Cats vs Dogs)
- Fine-tuned at low learning rate

---

## 📊 Results Summary
| Dataset       | Model Type        | Accuracy |
|---------------|-------------------|----------|
| MNIST         | Custom CNN        | ~98%     |
| CIFAR-10      | Custom CNN + Aug  | ~75%     |
| Cats vs Dogs  | MobileNetV2 TL    | ~84%     |

---

## 📷 Visualizations
- Training vs Validation Accuracy/Loss curves
- Confusion Matrices
- ROC Curves
- Sample Predictions

---

## 🚀 How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-recognition-bootcamp.git
```
2. Open in **Google Colab** or local Jupyter Notebook.
3. Install dependencies:
```bash
pip install tensorflow keras matplotlib seaborn scikit-learn
```
4. Run the notebooks in order:
   - `Day01_Preprocessing_and_Exploration.ipynb`
   - `Day02_CNN_Model_Training.ipynb`
   - `Day03_Augmentation_and_Evaluation.ipynb`
   - `Day04_Transfer_Learning.ipynb`
   - `Day05_Prediction_and_Deployment.ipynb`

---

## 📦 Deployment
- Saved model in `.h5` format for reuse.
- Demonstrated loading the model and predicting on new data.
- Potential deployment targets: Flask API, Streamlit app, or TensorFlow Lite for mobile.

---

## 🎯 Key Takeaways
- CNNs are powerful for feature extraction and classification in images.
- Data augmentation boosts model robustness.
- Transfer learning accelerates training and improves accuracy with smaller datasets.
- Proper evaluation and visualization ensure reliable model performance.
- Documentation and clean repo structure make projects portfolio-ready.

---

## 📹 Demo
*(Add link to your 30-second prediction demo video)*

---

## 📝 Author
**Srinivasan Sriram Tirunelvelli**  

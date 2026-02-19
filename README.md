# 🌸 Iris Classification: ML vs Deep Learning

A Streamlit web application that compares a traditional Machine Learning model (Perceptron) with a Deep Learning model (Artificial Neural Network) on the Iris dataset.

---

## 🚀 Live Features

- 🔮 Manual prediction using sliders  
- 📂 Batch prediction via CSV upload  
- 📊 Model accuracy comparison  
- 📈 Confusion matrix visualization  
- 📊 Pairplot visualization  
- 🔥 Correlation heatmap  
- 📥 Download predictions as CSV  

---

## 📊 Dataset

The project uses the **Iris Dataset**, a classic multi-class classification dataset.

- Total Samples: 150
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Classes:
  - Setosa
  - Versicolor
  - Virginica

---

## ⚙️ Data Preprocessing

- Train-Test Split
- Feature Scaling using `StandardScaler`
- Label Encoding
- One-Hot Encoding (for ANN)

---

## 🤖 Models Used

### 🔹 Perceptron (Scikit-learn)

- Linear classifier
- Learns a linear decision boundary
- Suitable for linearly separable data

### 🔹 Artificial Neural Network (TensorFlow / Keras)

- Fully connected dense network
- Hidden Layer Activation: ReLU
- Output Layer Activation: Softmax
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Target Encoding: One-Hot Encoding

---

## 🏗 ANN Architecture

# 🌸 Iris Classification: ML vs Deep Learning

A Streamlit web application that compares a traditional Machine Learning model (Perceptron) with a Deep Learning model (Artificial Neural Network) on the Iris dataset.

---

## 🚀 Live Features

- 🔮 Manual prediction using sliders  
- 📂 Batch prediction via CSV upload  
- 📊 Model accuracy comparison  
- 📈 Confusion matrix visualization  
- 📊 Pairplot visualization  
- 🔥 Correlation heatmap  
- 📥 Download predictions as CSV  

---

## 📊 Dataset

The project uses the **Iris Dataset**, a classic multi-class classification dataset.

- Total Samples: 150
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Classes:
  - Setosa
  - Versicolor
  - Virginica

---

## ⚙️ Data Preprocessing

- Train-Test Split
- Feature Scaling using `StandardScaler`
- Label Encoding
- One-Hot Encoding (for ANN)

---

## 🤖 Models Used

### 🔹 Perceptron (Scikit-learn)

- Linear classifier
- Learns a linear decision boundary
- Suitable for linearly separable data

### 🔹 Artificial Neural Network (TensorFlow / Keras)

- Fully connected dense network
- Hidden Layer Activation: ReLU
- Output Layer Activation: Softmax
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Target Encoding: One-Hot Encoding

---

## 🏗 ANN Architecture

- Input Layer: 4 neurons
- Hidden Layer: Dense (ReLU)
- Output Layer: 3 neurons (Softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy


---

## 📊 Model Performance

The ANN model performs better than the Perceptron because it captures non-linear patterns in the data.

Confusion matrices are included in the app for visual comparison.

---

## 🛠 Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Seaborn
- Matplotlib
- Joblib

---

## 📂 Project Structure

├── app.py
├── Iris.csv
├── perceptron_model.pkl
├── ann_model.h5
├── scaler.pkl
├── label_encoder.pkl
├── results.json
├── confusion_perceptron.png
├── confusion_ann.png
└── requirements.txt

## ⭐ If You Like This Project

Give it a ⭐ on GitHub!

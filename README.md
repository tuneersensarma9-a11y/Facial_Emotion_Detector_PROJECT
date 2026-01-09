# 🎭 Real-Time Facial Emotion Recognition System

**Using CNN, TensorFlow, Keras & OpenCV**

## 📌 Overview

This project implements a **Real-Time Facial Emotion Recognition System** using **Deep Learning** and **Computer Vision** techniques.
It detects human faces from a live webcam feed and predicts the corresponding **facial emotion in real time**.

The system is designed as an **end-to-end machine learning project**, covering:

* Data preprocessing
* CNN model training
* Model evaluation
* Real-time inference using OpenCV

---

## 🚀 Key Features

* Real-time emotion detection via webcam
* Face detection using **Haar Cascade Classifier**
* Emotion classification using a **Convolutional Neural Network (CNN)**
* Supports **7 emotion classes**:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise
* Modular and easy-to-run codebase
* Suitable for academic, research, and internship evaluation

---

## 🧠 Model Information

* **Input:** 48 × 48 grayscale face images
* **Model Type:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow & Keras
* **Training:** Custom training pipeline using Jupyter Notebook

### ⚠️ Note on Model File Size

The trained model file (`.keras`) is **not included directly in this repository** due to GitHub file size limitations when uploading via the web interface.

---

## 🔽 Trained Model (Download Options)

You can obtain the trained model in **the following way**:

### : Train the Model Yourself (Recommended)

Run the training notebook:

```
trainmodel.ipynb
```

This will generate the trained `.keras` model locally.


## 🗂️ Project Structure

```
├── images/                 # Sample images / screenshots
├── realtimedetection.py    # Main script for real-time emotion detection
├── trainmodel.ipynb        # Model training notebook
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files & folders
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python realtimedetection.py
```

* Ensure your webcam is connected
* Press **ESC** to exit the application

---

## 🖥️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **OpenCV**
* **NumPy**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📊 Applications

* Human-Computer Interaction (HCI)
* Emotion-aware AI systems
* Computer Vision research
* Academic mini & major projects
* Internship and placement portfolios

---

## 🔮 Future Enhancements

* Improve accuracy using larger datasets
* Add GUI or web-based interface
* Multi-face emotion detection
* Deploy as a web or mobile application
* Replace Haar Cascade with deep face detectors (MTCNN / DNN)

---

## 📜 Disclaimer

This project is intended **for educational and research purposes only**.
Emotion predictions may vary depending on lighting, pose, and image quality.

---

## 👤 Author

**Tuneer Sen Sarma**
GitHub: [https://github.com/tuneersensarma9-a11y](https://github.com/tuneersensarma9-a11y)

---

⭐ If you find this project useful, consider giving it a **star**!

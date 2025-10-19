# 🌍 CBC_A-014: NDVI Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

**A machine learning-powered web application for predicting Normalized Difference Vegetation Index (NDVI) values across different geographical regions.**

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Project Purpose](#-project-purpose)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Introduction

The **CBC_A-014 NDVI Prediction System** is an intelligent web application designed to predict vegetation health and density using the Normalized Difference Vegetation Index (NDVI). This project leverages deep learning models to provide accurate NDVI predictions for different geographical zones (North, South, East, and West), making it a valuable tool for environmental monitoring, agriculture, and land management.

---

## 🎯 Project Purpose

NDVI is a crucial metric in remote sensing and environmental science, helping researchers and practitioners:

- **Monitor Vegetation Health**: Track the vitality and density of vegetation over time
- **Agricultural Planning**: Support precision agriculture and crop management decisions
- **Environmental Assessment**: Evaluate ecosystem changes and land use patterns
- **Climate Research**: Understand the impact of climate variations on vegetation

This system democratizes access to NDVI predictions by providing an intuitive web interface backed by machine learning models trained for regional accuracy.

---

## ✨ Features

- 🗺️ **Regional Predictions**: Separate models for North, South, East, and West zones for enhanced accuracy
- 🤖 **Deep Learning Models**: Utilizes trained neural network models (.h5 format) for reliable predictions
- 🌐 **Web-Based Interface**: User-friendly HTML interface for easy interaction
- ⚡ **Fast Processing**: Quick prediction response times through optimized Flask backend
- 📊 **Scalable Architecture**: Modular design allows for easy expansion and model updates

---

## 🛠 Tech Stack

### Backend
- **Python 3.x**: Core programming language
- **Flask**: Lightweight web framework for API and routing
- **TensorFlow/Keras**: Deep learning framework for model inference
- **NumPy**: Numerical computing for data processing

### Frontend
- **HTML5**: Markup structure
- **CSS3**: Styling (integrated in templates)
- **JavaScript**: Client-side interactivity

### Machine Learning
- **Neural Networks**: Custom-trained models for each geographical zone
- **Model Format**: HDF5 (.h5) for efficient model storage and loading

---

## 📦 Installation & Setup

### Prerequisites

```bash
# Python 3.7 or higher
python --version

# pip package manager
pip --version
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sachethanv/CBC_A-014.git
cd CBC_A-014
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install flask tensorflow numpy
```

### Step 4: Verify Model Files

Ensure the following model files are present in the project directory:
- `NDVI predictor model east.h5`
- `NDVI predictor model north.h5`
- `NDVI predictor model south.h5`
- `NDVI predictor model west.h5`

---

## 🚀 Usage

### Running the Application

1. **Start the Flask server:**

```bash
python app.py
```

2. **Access the web interface:**

Open your browser and navigate to:
```
http://localhost:5000
```

3. **Make Predictions:**
   - Select your geographical region (North, South, East, or West)
   - Input the required parameters
   - Click "Predict" to receive NDVI values
   - View and analyze the results

### API Endpoints

The application exposes RESTful API endpoints for programmatic access:

```bash
# Example prediction request
POST /predict
Content-Type: application/json

{
  "region": "north",
  "parameters": [...]
}
```

---

## 🧠 Model Architecture

The system uses four independently trained neural network models:

- **North Model**: Optimized for northern geographical patterns
- **South Model**: Tailored for southern vegetation characteristics
- **East Model**: Specialized for eastern regional data
- **West Model**: Configured for western zone predictions

Each model has been trained on region-specific datasets to ensure maximum accuracy and reliability.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute this software in accordance with the license terms.

```
MIT License

Copyright (c) 2025 Sachethan V

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Sachethan V**

- 🐙 GitHub: [@Sachethanv](https://github.com/Sachethanv)
- 📬 For questions, suggestions, or collaboration opportunities, please open an issue on GitHub

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ for environmental science and sustainable development**

</div>

---

# 🐾 Cats vs Dogs Classifier

This machine learning project uses a **Convolutional Neural Network (CNN)** to classify images as either **cats** or **dogs**. Built using the popular **Kaggle "Dogs vs Cats" dataset**, the model handles data preprocessing, augmentation, training, and evaluation—all within an interactive Jupyter Notebook.

---

## 👨‍💻 Author

**Krish Kumar**
GitHub: [@krisjscott](https://github.com/krisjscott)

---

## 🐶🛠️ Requirements

* Python 3.7 or higher
* Jupyter Notebook or VS Code

### 🧪 Key Libraries

```
tensorflow
keras
numpy
opencv-python or Pillow
matplotlib
```

> You can use `torch` instead of `tensorflow` if modifying the model to PyTorch.

---

## ✨ Features

* 🗂️ **Preprocesses** image dataset (resizing, normalization, augmentation)
* 🧱 Builds a CNN with convolutional, pooling, and dense layers
* 🏋️ **Trains** the model with real-time accuracy and loss tracking
* 📈 **Evaluates** performance using validation metrics
* 🐕🐈 **Predicts** on new images via notebook interface

---

## 🚀 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/krisjscott/animal-prediction.git
cd cats-vs-dogs
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is missing, manually install:*

```bash
pip install tensorflow keras numpy opencv-python matplotlib
```

---

### 3. Download & prepare the dataset

* Download the **Dogs vs Cats** dataset from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats).
* Unzip and structure it like this:

```
data/
  train/
    cats/
    dogs/
```

#### 📦 If using the Kaggle CLI:

```bash
kaggle competitions download -c dogs-vs-cats -p data
unzip data/dogs-vs-cats.zip -d data
unzip data/train.zip -d data/train
```

---

### 4. Launch Jupyter Notebook or VS Code

```bash
jupyter notebook
```

* Open `CatsvsDogs_prediction_CNN.ipynb`
* Follow the cells to:

  * Load and preprocess the data
  * Train the CNN
  * Evaluate performance
  * Predict on new images

---

## 🧪 Example: Predicting a New Image

Inside the notebook:

```python
img = preprocess_image('my_cat_or_dog.jpg')
pred = model.predict(img)
print("Predicted:", "Dog" if pred > 0.5 else "Cat")
```

---

## 🪟 Preview

![Screenshot](https://github.com/user-attachments/assets/2f2ec2e1-97ae-4db0-8ddb-fcca989fe3fb)

---

## 📊 Sample Results

* **Training Accuracy**: \~95%
* **Validation Accuracy**: \~90%
* Includes visualization of accuracy/loss curves
* Shows sample predictions during inference

---

## 🤝 Contributing

Pull requests are welcome! You can contribute by:

* Improving model performance
* Adding transfer learning (e.g., VGG16, ResNet)
* Creating a GUI or web app interface
* Enhancing documentation or tutorials

---

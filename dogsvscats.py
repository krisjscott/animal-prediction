import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


DATADIR = r"C:\\Users\\thekr\\OneDrive\\Documents\\program\\python\\dogs_vs_cats\\train\\train"
CATEGORIES = ["cat", "dog"]
IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 3000

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

def collect_image_paths(datadir, categories, max_per_class):
    image_paths, labels = [], []
    count = {cat: 0 for cat in categories}
    
    for file in tqdm(os.listdir(datadir), desc="Collecting file paths"):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        for idx, cat in enumerate(categories):
            if cat in file.lower() and count[cat] < max_per_class:
                image_paths.append(os.path.join(datadir, file))
                labels.append(idx)
                count[cat] += 1
        if all(count[c] >= max_per_class for c in categories):
            break

    print(f"Collected {len(image_paths)} images")
    return image_paths, labels


def load_and_preprocess_images(paths):
    images = []
    for path in tqdm(paths, desc="Loading and preprocessing images"):
        img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        images.append(img_array)
    images = np.array(images)
    return preprocess_input(images)

def predict_image(image_path, model, base_model, categories):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    features = base_model.predict(img_array)
    features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    print(f"\ Prediction: {categories[prediction].upper()}")

def main():
    image_paths, labels = collect_image_paths(DATADIR, CATEGORIES, MAX_IMAGES_PER_CLASS)
    X = load_and_preprocess_images(image_paths)
    y = np.array(labels)

    features = base_model.predict(X, batch_size=32, verbose=1)
    features = features.reshape(features.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    svm_model = SVC(kernel='rbf', C=1.0)
    svm_model.fit(X_train, y_train)


    y_pred = svm_model.predict(X_test)
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print(" Classification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))

    test_image_path= r"C:\\Users\\thekr\\OneDrive\\Documents\\program\\python\\dogs_vs_cats\\test1\\test1\\12487.jpg"

    predict_image(test_image_path, svm_model, base_model, CATEGORIES)

if __name__ == "__main__":
    main()






import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img.flatten()
            images.append(img)
            labels.append(label)
        else:
            print(f"Warning: Unable to load image {img_path}")
    return images, labels

# Load brain and lung MRI images
brain_normal_images, brain_normal_labels = load_images_from_folder('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\BRAIN\\no', 0)
brain_tumor_images, brain_tumor_labels = load_images_from_folder('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\BRAIN\\yes', 1)

lung_normal_images, lung_normal_labels = load_images_from_folder('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\LUNGS\\no', 2)
lung_tumor_images, lung_tumor_labels = load_images_from_folder('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\LUNGS\\yes', 3)

# Combine the data for SVM
images = np.array(brain_normal_images + brain_tumor_images + lung_normal_images + lung_tumor_images)
labels = np.array(brain_normal_labels + brain_tumor_labels + lung_normal_labels + lung_tumor_labels)

# Check if images and labels are loaded correctly
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images found. Please check the image paths and folders.")

# Split the data into training and testing sets for SVM
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the SVM model to classify brain vs lung
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions with SVM
y_pred = svm_model.predict(X_test)

def classify_image_type(image_path, svm_model):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = img.flatten()
        img = np.expand_dims(img, axis=0)
        prediction = svm_model.predict(img)
        if prediction[0] == 0 or prediction[0] == 1:
            return 'brain'
        elif prediction[0] == 2 or prediction[0] == 3:
            return 'lung'
    else:
        raise ValueError(f"Unable to load image {image_path}")

# Calculate accuracy and precision for SVM
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
print(f'SVM Accuracy: {accuracy * 100:.2f}%')
print(f'SVM Precision: {precision * 100:.2f}%')

# Define the CNN model architecture
def create_cnn_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (tumor or no tumor)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and save CNN model for brain images
def train_brain_cnn():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\BRAIN', target_size=(150, 150), batch_size=32, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\BRAIN', target_size=(150, 150), batch_size=32, class_mode='binary')

    input_shape = (150, 150, 3)
    brain_model = create_cnn_model(input_shape)
    brain_model.fit(train_generator, epochs=10, validation_data=validation_generator)
    brain_model.save('brain_tumor_detection_model.h5')

# Train and save CNN model for lung images
def train_lung_cnn():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\LUNGS', target_size=(150, 150), batch_size=32, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory('C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\LUNGS', target_size=(150, 150), batch_size=32, class_mode='binary')

    input_shape = (150, 150, 3)
    lung_model = create_cnn_model(input_shape)
    lung_model.fit(train_generator, epochs=10, validation_data=validation_generator)
    lung_model.save('lung_tumor_detection_model.h5')

# Train the CNN models
train_brain_cnn()
train_lung_cnn()

# Load the CNN models for tumor detection
brain_cnn_model = load_model('brain_tumor_detection_model.h5')
lung_cnn_model = load_model('lung_tumor_detection_model.h5')

# Function to classify whether the brain or lung image has a tumor
def classify_tumor(image_path, image_type):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        if image_type == 'brain':
            prediction = brain_cnn_model.predict(img)
            return 'tumor' if prediction[0] > 0.7 else 'no tumor'
        elif image_type == 'lung':
            prediction = lung_cnn_model.predict(img)
            return 'tumor' if prediction[0] > 0.7 else 'no tumor'
    else:
        raise ValueError(f"Unable to load image {image_path}")

# Test the combined classifier with a new image
test_image_path = "C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\BRAIN\\yes\\Tr-me_0032.jpg"
image_type = classify_image_type(test_image_path, svm_model)
print(f'The image is classified as: {image_type}')
tumor_status = classify_tumor(test_image_path, image_type)
print(f'The image is classified as: {image_type} with {tumor_status}')

# Evaluate CNN models
def evaluate_cnn_model(model_path, directory_path):
    model = load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(directory_path, target_size=(150, 150), batch_size=32, class_mode='binary', shuffle=False)

    loss, accuracy = model.evaluate(test_generator)
    y_true = test_generator.classes
    y_pred = (model.predict(test_generator) > 0.5).astype("int32").flatten()

    precision = precision_score(y_true, y_pred, average='weighted')

    return accuracy, precision

# Evaluate CNN models
brain_accuracy, brain_precision = evaluate_cnn_model('brain_tumor_detection_model.h5', 'C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\BRAIN')
lung_accuracy, lung_precision = evaluate_cnn_model('lung_tumor_detection_model.h5', 'C:\\Users\\krahu\\OneDrive\\Desktop\\Project-1\\lungsandbrain\\LUNGS')

print(f'Brain CNN Accuracy: {brain_accuracy * 100:.2f}%')
print(f'Brain CNN Precision: {brain_precision * 100:.2f}%')

print(f'Lung CNN Accuracy: {lung_accuracy * 100:.2f}%')
print(f'Lung CNN Precision: {lung_precision * 100:.2f}%')
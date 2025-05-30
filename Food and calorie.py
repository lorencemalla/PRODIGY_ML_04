import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set path to the training dataset
dataset_dir = "D:Internship (Prodigy Infotech)/dataset/archive/training"

print("Loading and preprocessing images...")

data = []
labels = []

# Load and process images
for category in os.listdir(dataset_dir):
    folder = os.path.join(dataset_dir, category)
    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            img_flat = img.flatten()
            data.append(img_flat)
            labels.append(category)
        except Exception:
            continue

print(f"Total images loaded: {len(data)}")

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Save label encoder for future use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train, y_train)
print("Training complete.")

# Save model
with open("food_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Show a sample prediction
sample = random.randint(0, len(X_test) - 1)
img = np.array(X_test[sample]).reshape(100, 100)
predicted_label = le.inverse_transform([y_pred[sample]])[0]

plt.imshow(img, cmap="gray")
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()

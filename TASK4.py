import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==========================================
# LOAD AND PREPARE DATA
# ==========================================
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape data to include the channel dimension (needed for Conv2D)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==========================================
# DEFINE MODEL
# ==========================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# ========================================== 
# Compile the model with an optimizer, loss function, and metrics.
# This part was missing/incorrectly formatted in the original cell.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
    metrics=['accuracy']
)

# ==========================================
# MODEL SUMMARY
# ==========================================
model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# ==========================================
# EVALUATE MODEL
# ==========================================
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy)

# ==========================================
# PLOT ACCURACY GRAPH
# ==========================================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# ==========================================
# PLOT LOSS GRAPH
# ==========================================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# ========================================== 
# MAKE PREDICTIONS
# ==========================================
predictions = model.predict(x_test)

# Show prediction for first test image
# Reshape x_test[0] to 2D for imshow
plt.imshow(x_test[0].reshape(28, 28), cmap='gray') 
plt.title(f"Predicted: {np.argmax(predictions[0])}")
plt.axis('off')
plt.show()

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ==========================================
# SAVE MODEL
# ==========================================
model.save('mnist_cnn_model.h5')

print("\nModel saved successfully!")
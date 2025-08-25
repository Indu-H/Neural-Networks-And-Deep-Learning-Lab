# Define OR and XOR datasets
import numpy as np
X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_OR = np.array([[0], [1], [1], [1]], dtype=np.float32)
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_XOR = np.array([[0], [1], [1], [0]], dtype=np.float32)

import tensorflow as tf
# Function to create and train a single-layer perceptron with improved accuracy
def train_perceptron(X, y, epochs=100, learning_rate=0.1):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=0)
    return model
# Train on OR gate with improved accuracy
model_OR = train_perceptron(X_OR, y_OR, epochs=500, learning_rate=0.5)
# Evaluate on OR
loss_OR, accuracy_OR = model_OR.evaluate(X_OR, y_OR)
print(f"OR Gate Accuracy: {accuracy_OR}")
# Train on XOR gate with improved accuracy (not possible with a single-layer perceptron)
# However, we can try increasing epochs and learning rate
model_XOR = train_perceptron(X_XOR, y_XOR, epochs=1000, learning_rate=0.8)
# Evaluate on XOR
loss_XOR, accuracy_XOR = model_XOR.evaluate(X_XOR, y_XOR)
print(f"XOR Gate Accuracy: {accuracy_XOR}")

#Make a prediction using model_OR
input1 = 0
input2 = 0
user_input = np.array([[input1, input2]])
prediction = model_XOR.predict(user_input)
if prediction > 0.5:
    print("The model predicts 1 for your input.")
else:
    print("The model predicts 0 for your input.")

#output
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 260ms/step - accuracy: 1.0000 - loss: 0.0383
# OR Gate Accuracy: 1.0
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 224ms/step - accuracy: 0.2500 - loss: 0.6931
# XOR Gate Accuracy: 0.25
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step
# The model predicts 0 for your input.

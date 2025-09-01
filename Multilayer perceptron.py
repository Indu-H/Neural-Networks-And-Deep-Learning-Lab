 import tensorflow as tf
 from tensorflow.keras.datasets import fashion_mnist
 (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
 X_train, X_test = X_train / 255.0, X_test / 255.0
 model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
 ])
 model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
metrics=['accuracy'])
 lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 
* 10 ** (epoch / 20))
 history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, 
y_test), callbacks=[lr_schedule])
 import matplotlib.pyplot as plt
 plt.plot(history.history['accuracy'], label='Training Accuracy')
 plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy')
 plt.legend()
 plt.show()


#output
# Epoch 1/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 11s 5ms/step - accuracy: 0.7822 - loss: 0.6140 - val_accuracy: 0.8299 - val_loss: 0.4626 - learning_rate: 0.0010
# Epoch 2/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8207 - loss: 0.5029 - val_accuracy: 0.8124 - val_loss: 0.4798 - learning_rate: 0.0011
# Epoch 3/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8280 - loss: 0.4813 - val_accuracy: 0.8392 - val_loss: 0.4448 - learning_rate: 0.0013
# Epoch 4/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8302 - loss: 0.4731 - val_accuracy: 0.8389 - val_loss: 0.4327 - learning_rate: 0.0014
# Epoch 5/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8346 - loss: 0.4619 - val_accuracy: 0.8567 - val_loss: 0.3943 - learning_rate: 0.0016
# Epoch 6/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8377 - loss: 0.4461 - val_accuracy: 0.8547 - val_loss: 0.4000 - learning_rate: 0.0018
# Epoch 7/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8399 - loss: 0.4476 - val_accuracy: 0.8518 - val_loss: 0.4099 - learning_rate: 0.0020
# Epoch 8/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8389 - loss: 0.4454 - val_accuracy: 0.8458 - val_loss: 0.4162 - learning_rate: 0.0022
# Epoch 9/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8386 - loss: 0.4498 - val_accuracy: 0.8484 - val_loss: 0.4304 - learning_rate: 0.0025
# Epoch 10/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8402 - loss: 0.4449 - val_accuracy: 0.8597 - val_loss: 0.3898 - learning_rate: 0.0028

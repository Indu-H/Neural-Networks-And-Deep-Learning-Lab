 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.datasets import cifar10
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
 from sklearn.metrics import accuracy_score, precision_score, recall_score
 # Load CIFAR-10 dataset
 (X_train, y_train), (X_test, y_test) = cifar10.load_data()
 # Normalize pixel values to be between 0 and 1
 X_train = X_train.astype('float32') / 255.0
 X_test = X_test.astype('float32') / 255.0
 # Convert class vectors to binary class matrices (one-hot encoding)
 y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
 y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
 # Define the CNN model
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
input_shape=(32, 32, 3)))
 model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
 model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dense(10, activation='softmax'))
 # Compile the model
 model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
 model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
 # Evaluate the model
 loss, accuracy = model.evaluate(X_test, y_test)
 print('Test accuracy:', accuracy)
 # Make predictions on the test set
 y_pred_prob = model.predict(X_test)
 y_pred = np.argmax(y_pred_prob, axis=1)
 y_true = np.argmax(y_test, axis=1)
 # Calculate precision and recall
 precision = precision_score(y_true, y_pred, average='macro')
 recall = recall_score(y_true, y_pred, average='macro')
 print('Precision:', precision)
 print('Recall:', recall)


# output
# Epoch 1/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 90s 110ms/step - accuracy: 0.4446 - loss: 1.5077 - val_accuracy: 0.5421 - val_loss: 1.2967
# Epoch 2/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 86s 110ms/step - accuracy: 0.6482 - loss: 0.9935 - val_accuracy: 0.6724 - val_loss: 0.9218
# Epoch 3/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 85s 109ms/step - accuracy: 0.7208 - loss: 0.7927 - val_accuracy: 0.7313 - val_loss: 0.7777
# Epoch 4/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 87s 111ms/step - accuracy: 0.7708 - loss: 0.6513 - val_accuracy: 0.7443 - val_loss: 0.7476
# Epoch 5/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 87s 111ms/step - accuracy: 0.8096 - loss: 0.5459 - val_accuracy: 0.7698 - val_loss: 0.6909
# Epoch 6/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 90s 115ms/step - accuracy: 0.8395 - loss: 0.4596 - val_accuracy: 0.7726 - val_loss: 0.6820
# Epoch 7/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 88s 112ms/step - accuracy: 0.8661 - loss: 0.3782 - val_accuracy: 0.7706 - val_loss: 0.7185
# Epoch 8/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 86s 110ms/step - accuracy: 0.8885 - loss: 0.3127 - val_accuracy: 0.7626 - val_loss: 0.8002
# Epoch 9/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 86s 111ms/step - accuracy: 0.9095 - loss: 0.2563 - val_accuracy: 0.7778 - val_loss: 0.7988
# Epoch 10/10
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 87s 111ms/step - accuracy: 0.9271 - loss: 0.2058 - val_accuracy: 0.7719 - val_loss: 0.9225
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - accuracy: 0.7719 - loss: 0.9225
# Test accuracy: 0.7718999981880188
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step
# Precision: 0.7709879825956258
# Recall: 0.7719

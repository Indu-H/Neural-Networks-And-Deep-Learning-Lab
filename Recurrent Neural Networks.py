from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
 # Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
 # Pad sequences to ensure equal length inputs
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)
model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=200),tf.keras.layers.LSTM(128, return_sequences=False),tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
import matplotlib.pyplot as plt
 # Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score
 # Get predictions and round them to nearest integer
y_pred = (model.predict(X_test) > 0.5).astype("int32")
 # Evaluate performance
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
def preprocess_input(text):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts([text])  # Fit only on the user input
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence
 # Get user input
user_input = input("Enter a movie review: ")
 # Preprocess the user input
processed_input = preprocess_input(user_input)
 # Make prediction
prediction = model.predict(processed_input)
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
 # Output the result
print(f"Predicted Sentiment: {sentiment} (Probability: {prediction[0][0]:.2f})")


# output
# Epoch 1/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 141s 351ms/step - accuracy: 0.7753 - loss: 0.4605 - val_accuracy: 0.8580 - val_loss: 0.3467
# Epoch 2/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 136s 349ms/step - accuracy: 0.8930 - loss: 0.2714 - val_accuracy: 0.8671 - val_loss: 0.3335
# Epoch 3/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 138s 353ms/step - accuracy: 0.9230 - loss: 0.2007 - val_accuracy: 0.8634 - val_loss: 0.3425
# Epoch 4/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 137s 349ms/step - accuracy: 0.9451 - loss: 0.1505 - val_accuracy: 0.8603 - val_loss: 0.3500
# Epoch 5/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 140s 357ms/step - accuracy: 0.9619 - loss: 0.1077 - val_accuracy: 0.8632 - val_loss: 0.4222
# Epoch 6/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 139s 354ms/step - accuracy: 0.9716 - loss: 0.0814 - val_accuracy: 0.8539 - val_loss: 0.4509
# Epoch 7/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 140s 359ms/step - accuracy: 0.9790 - loss: 0.0629 - val_accuracy: 0.8495 - val_loss: 0.6055
# Epoch 8/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 139s 355ms/step - accuracy: 0.9792 - loss: 0.0597 - val_accuracy: 0.8486 - val_loss: 0.5543
# Epoch 9/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 142s 364ms/step - accuracy: 0.9848 - loss: 0.0465 - val_accuracy: 0.8565 - val_loss: 0.6013
# Epoch 10/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 143s 367ms/step - accuracy: 0.9882 - loss: 0.0357 - val_accuracy: 0.8463 - val_loss: 0.6264

# 388/782 ━━━━━━━━━━━━━━━━━━━━ 27s 69ms/step





import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
Y_train = 3 * X_train + 2

model = Sequential([
    Dense(units=16, input_shape=[1], activation='relu'), 
    Dense(units=8, activation='relu'),
    Dense(units=1) 
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("Training the model...")
history = model.fit(X_train, Y_train, epochs=500, verbose=0)
print("Model trained.")

X_test = np.array([5, 3.3], dtype=float)
predictions = model.predict(X_test)

print("\nPredictions:")
for i, x in enumerate(X_test):
    print(f"For X = {x}: Y ≈ {predictions[i][0]:.2f}")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

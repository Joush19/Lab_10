import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)  
Y_train = 3 * X_train + 2  

model = Sequential([
    Dense(units=1, input_shape=[1])  
])

model.compile(optimizer='sgd', loss='mean_squared_error')  

print("Entrenando el modelo...")
history = model.fit(X_train, Y_train, epochs=500, verbose=0) 
print("Modelo entrenado con éxito.")

X_test = np.array([5, 3.3], dtype=float)
predictions = model.predict(X_test)

print("\nResultados de las predicciones:")
for i, x in enumerate(X_test):
    print(f"Para X = {x}: Y ≈ {predictions[i][0]:.2f}")

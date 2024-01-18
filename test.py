import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def linear_shift_invariance(filter_coefficients, input_signal):
    filter_coefficients = filter_coefficients[::-1, ::-1]
    output = np.zeros_like(input_signal, dtype=np.float64)
    
    for i in range(len(input_signal)):
        for j in range(len(input_signal[0])):
            for m, k in enumerate(range(i-1, i+2)):
                for n, l in enumerate(range(j-1, j+2)):
                    if k >= 0 and l >= 0 and k < len(input_signal) and l < len(input_signal[0]):
                        output[i, j] += input_signal[k, l] * filter_coefficients[m, n]
    return output

# Generate random data for demonstration
np.random.seed(42)
X = np.random.random((1000, 4, 4))  # 1000 samples of 4x4 matrices
y = np.array([linear_shift_invariance(np.random.random((3, 3)), x) for x in X])

# Reshape the actual outputs to match the model's output shape
y = y.reshape((y.shape[0], -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4, 4)),  # Flatten the 4x4 input matrix
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(9)  # Output layer with 9 neurons for the 3x3 output
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Use the trained model to predict the output for a new input matrix
new_input_matrix = np.random.random((1, 4, 4))
predicted_output = model.predict(new_input_matrix)

# Display the results
print("Input Matrix:")
print(new_input_matrix.squeeze())
print("\nPredicted Output Matrix:")
print(predicted_output.reshape((3, 3)))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# --- 1. Load Data ---
print("Loading dataset...")
df = pd.read_csv('orbit_training_data.csv')
df = df.dropna()

X = df[['angle', 'duration', 'power', 'mass']].values
Y_reg = df[['apoapsis', 'periapsis']].values    # Regression targets (m)
Y_clf = df['stable'].values                     # Classification target

# --- 2. Preprocessing & Data Splitting ---
print("Preprocessing and scaling datasets...")
X_train, X_test, Y_reg_train, Y_reg_test, Y_clf_train, Y_clf_test = train_test_split(
    X, Y_reg, Y_clf, test_size=0.2, random_state=42
)

# Scale inputs (Features)
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# CRITICAL FIX: Scale outputs to values between 0 and 1 to prevent exploding MSE losses
target_scaler = MinMaxScaler()
Y_reg_train_scaled = target_scaler.fit_transform(Y_reg_train)
Y_reg_test_scaled = target_scaler.transform(Y_reg_test)

# Export both scalers inside a single dictionary file for your app pipeline
scalers = {
    'feature_scaler': feature_scaler,
    'target_scaler': target_scaler
}
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scalers, f)

# --- 3. Build Multi-Output Neural Network (Functional API) ---
print("Configuring multi-headed TensorFlow layers...")

input_layer = layers.Input(shape=(4,), name='rocket_parameters')
shared_dense1 = layers.Dense(64, activation='relu')(input_layer)
shared_dense2 = layers.Dense(64, activation='relu')(shared_dense1)

# Branch A: Regression Head (Now outputs values scaled between 0 and 1)
reg_dense = layers.Dense(32, activation='relu')(shared_dense2)
reg_output = layers.Dense(2, activation='linear', name='trajectory_output')(reg_dense)

# Branch B: Classification Head (Outputs binary probability score)
clf_dense = layers.Dense(16, activation='relu')(shared_dense2)
clf_output = layers.Dense(1, activation='sigmoid', name='stability_output')(clf_dense)

# Merge layers into a unified structural model entity
model = models.Model(inputs=input_layer, outputs=[reg_output, clf_output])

# Compile using specialized losses
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'trajectory_output': 'mse', 
        'stability_output': 'binary_crossentropy'
    },
    metrics={
        'stability_output': 'accuracy'
    }
)

# --- 4. Train Neural Engine ---
print("Training Network Model (30 epochs)...")
model.fit(
    X_train_scaled, 
    {'trajectory_output': Y_reg_train_scaled, 'stability_output': Y_clf_train},
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- 5. Export Functional Network File ---
model.save('orbit_predictor_model.h5')
print("Model training complete. Non-corrupted 'orbit_predictor_model.h5' generated successfully.")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- 1. Load Data ---
print("Loading dataset...")
df = pd.read_csv('orbit_training_data.csv')

X = df[['angle', 'duration', 'power', 'mass']].values
Y_reg = df[['apoapsis', 'periapsis']].values    # Regression targets
Y_clf = df['stable'].values                     # Classification target

# --- 2. Preprocessing & Data Splitting ---
print("Preprocessing data...")
X_train, X_test, Y_reg_train, Y_reg_test, Y_clf_train, Y_clf_test = train_test_split(
    X, Y_reg, Y_clf, test_size=0.2, random_state=42
)

# Standardize inputs to assist neural convergence gradients
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler binary object for use in runtime inference (app.py)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# --- 3. Build Multi-Output Neural Network (Functional API) ---
print("Building TensorFlow network architecture...")

input_layer = layers.Input(shape=(4,), name='rocket_parameters')
shared_dense1 = layers.Dense(64, activation='relu')(input_layer)
shared_dense2 = layers.Dense(64, activation='relu')(shared_dense1)

# Branch A: Regression Head (Predicts numeric Apoapsis & Periapsis heights)
reg_dense = layers.Dense(32, activation='relu')(shared_dense2)
reg_output = layers.Dense(2, name='trajectory_output')(reg_dense)

# Branch B: Classification Head (Predicts binary orbital stability probability)
clf_dense = layers.Dense(16, activation='relu')(shared_dense2)
clf_output = layers.Dense(1, activation='sigmoid', name='stability_output')(clf_dense)

# Unify architecture inputs and outputs into a single model block
model = models.Model(inputs=input_layer, outputs=[reg_output, clf_output])

# Compile using specialized losses for multi-output convergence
model.compile(
    optimizer='adam',
    loss={
        'trajectory_output': 'mse', 
        'stability_output': 'binary_crossentropy'
    },
    metrics={
        'stability_output': 'accuracy'
    }
)

# --- 4. Train Neural Engine ---
print("Training Neural Network (30 epochs)...")
model.fit(
    X_train_scaled, 
    {'trajectory_output': Y_reg_train, 'stability_output': Y_clf_train},
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- 5. Save Network Weights ---
model.save('orbit_predictor_model.h5')
print("Model training complete. File 'orbit_predictor_model.h5' generated.")

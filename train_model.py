import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- 1. Load Data & Fix Exploding Gradients ---
print("Loading dataset...")
df = pd.read_csv('orbit_training_data.csv')

# Drop any row that corrupted with bad simulation data
df = df.dropna()

X = df[['angle', 'duration', 'power', 'mass']].values

# CRITICAL FIX: Convert massive meter metrics into kilometers (e.g. 6,371,000m -> 6371km)
# This prevents the calculations from exploding into 'nan' values
Y_reg = df[['apoapsis', 'periapsis']].values / 1000.0  
Y_clf = df['stable'].values                     

# --- 2. Preprocessing & Data Splitting ---
print("Preprocessing and dividing data arrays...")
X_train, X_test, Y_reg_train, Y_reg_test, Y_clf_train, Y_clf_test = train_test_split(
    X, Y_reg, Y_clf, test_size=0.2, random_state=42
)

# Standardize feature metrics to assist neural optimization curves
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Export the active scalar model object for dashboard usage
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# --- 3. Build Multi-Output Neural Network (Functional API) ---
print("Configuring multi-headed TensorFlow layers...")

input_layer = layers.Input(shape=(4,), name='rocket_parameters')
shared_dense1 = layers.Dense(64, activation='relu')(input_layer)
shared_dense2 = layers.Dense(64, activation='relu')(shared_dense1)

# Branch A: Regression (Outputs numeric values in KM)
reg_dense = layers.Dense(32, activation='relu')(shared_dense2)
reg_output = layers.Dense(2, name='trajectory_output')(reg_dense)

# Branch B: Classification (Outputs binary probability score)
clf_dense = layers.Dense(16, activation='relu')(shared_dense2)
clf_output = layers.Dense(1, activation='sigmoid', name='stability_output')(clf_dense)

# Merge layers into unified structural entity
model = models.Model(inputs=input_layer, outputs=[reg_output, clf_output])

# Compile utilizing Adam optimizer with mean squared error mapping
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
print("Training Network Model (30 epochs)...")
model.fit(
    X_train_scaled, 
    {'trajectory_output': Y_reg_train, 'stability_output': Y_clf_train},
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- 5. Export Functional Network File ---
model.save('orbit_predictor_model.h5')
print("Model training complete. Non-corrupted 'orbit_predictor_model.h5' generated successfully.")

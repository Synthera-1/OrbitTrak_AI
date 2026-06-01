import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

print("Loading dataset...")
df = pd.read_csv('orbit_training_data.csv').dropna()

X = df[['angle', 'duration', 'power', 'mass']].values

# Convert to kilometers
apo_km = df['apoapsis'].values / 1000.0
peri_km = df['periapsis'].values / 1000.0

# CRITICAL STEP: Cap the maximum limit at 10,000 km to prevent astronomical spikes
apo_capped = np.clip(apo_km, 0.0, 10000.0)
peri_capped = np.clip(peri_km, 0.0, 10000.0)
raw_targets = np.column_stack((apo_capped, peri_capped))

X_train, X_test, Y_reg_train, Y_reg_test, Y_clf_train, Y_clf_test = train_test_split(
    X, raw_targets, df['stable'].values, test_size=0.2, random_state=42
)

# Scale the inputs (Sliders)
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale the outputs strictly between 0.0 and 1.0 using MinMaxScaler
target_scaler = MinMaxScaler()
Y_reg_train_scaled = target_scaler.fit_transform(Y_reg_train)
Y_reg_test_scaled = target_scaler.transform(Y_reg_test)

# Save both scalers into a single dictionary file for app.py
scalers = {
    'feature_scaler': feature_scaler,
    'target_scaler': target_scaler
}
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scalers, f)

# Build Model Architecture
input_layer = layers.Input(shape=(4,), name='rocket_parameters')
shared = layers.Dense(64, activation='relu')(input_layer)
shared = layers.Dense(64, activation='relu')(shared)

reg_head = layers.Dense(32, activation='relu')(shared)
reg_output = layers.Dense(2, activation='linear', name='trajectory_output')(reg_head)

clf_head = layers.Dense(16, activation='relu')(shared)
clf_output = layers.Dense(1, activation='sigmoid', name='stability_output')(clf_head)

model = models.Model(inputs=input_layer, outputs=[reg_output, clf_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'trajectory_output': 'mse', 'stability_output': 'binary_crossentropy'},
    metrics={'stability_output': 'accuracy'}
)

print("Training Network Model...")
model.fit(
    X_train_scaled, 
    {'trajectory_output': Y_reg_train_scaled, 'stability_output': Y_clf_train},
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

model.save('orbit_predictor_model.h5')
print("Success! Trained with standard MinMaxScaler.")

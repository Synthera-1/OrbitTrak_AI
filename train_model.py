import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- 1. Load Data & Logarithmic Scale Compression ---
print("Loading dataset...")
df = pd.read_csv('orbit_training_data.csv')
df = df.dropna()

X = df[['angle', 'duration', 'power', 'mass']].values

# Bound massive astronomical metrics into values under 10.0 using Log10
raw_apo = df['apoapsis'].values
raw_peri = df['periapsis'].values
log_apo = np.log10(np.clip(raw_apo, 1.0, None))
log_peri = np.log10(np.clip(raw_peri, 1.0, None))
Y_reg = np.column_stack((log_apo, log_peri))

Y_clf = df['stable'].values                     

# --- 2. Preprocessing & Data Splitting ---
X_train, X_test, Y_reg_train, Y_reg_test, Y_clf_train, Y_clf_test = train_test_split(
    X, Y_reg, Y_clf, test_size=0.2, random_state=42
)

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)

# --- 3. Build Model Architecture ---
input_layer = layers.Input(shape=(4,), name='rocket_parameters')
shared_dense1 = layers.Dense(64, activation='relu')(input_layer)
shared_dense2 = layers.Dense(64, activation='relu')(shared_dense1)

# Branch A: Regression Head (Linear outputs)
reg_dense = layers.Dense(32, activation='relu')(shared_dense2)
reg_output = layers.Dense(2, activation='linear', name='trajectory_output')(reg_dense)

# Branch B: Classification Head (Sigmoid binary outputs)
clf_dense = layers.Dense(16, activation='relu')(shared_dense2)
clf_output = layers.Dense(1, activation='sigmoid', name='stability_output')(clf_dense)

model = models.Model(inputs=input_layer, outputs=[reg_output, clf_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'trajectory_output': 'mse', 'stability_output': 'binary_crossentropy'},
    metrics={'stability_output': 'accuracy'}
)

# --- 4. Train Neural Engine ---
print("Training Network Model...")
model.fit(
    X_train_scaled, 
    {'trajectory_output': Y_reg_train, 'stability_output': Y_clf_train},
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# CRITICAL FORMAT UPDATE: Save as a native TensorFlow directory instead of an old .h5 file wrapper
model.save('orbit_predictor_model', save_format='tf')
print("Model training complete. Directory 'orbit_predictor_model' generated successfully.")

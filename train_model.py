import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("Loading dataset...")
df = pd.read_csv('orbit_training_data.csv').dropna()

X = df[['angle', 'duration', 'power', 'mass']].values

# ULTIMATE FIX: Convert meters to kilometers, then apply a smooth Log10 transformation
# This compresses numbers up to 100,000 km into a rock-solid scale between 0.0 and 6.0
apo_km = df['apoapsis'].values / 1000.0
peri_km = df['periapsis'].values / 1000.0

Y_reg = np.column_stack((
    np.log10(apo_km + 1.0),
    np.log10(peri_km + 1.0)
))
Y_clf = df['stable'].values                     

X_train, X_test, Y_reg_train, Y_reg_test, Y_clf_train, Y_clf_test = train_test_split(
    X, Y_reg, Y_clf, test_size=0.2, random_state=42
)

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)

# Network Architecture
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
    {'trajectory_output': Y_reg_train, 'stability_output': Y_clf_train},
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

model.save('orbit_predictor_model.h5')
print("Success! 'orbit_predictor_model.h5' generated with clean, numerical loss values.")

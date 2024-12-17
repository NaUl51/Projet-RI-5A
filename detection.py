import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import glob
import os


#IMPORT DATASET

# Chemin vers le dossier contenant les fichiers .h5
dataset_path = "./Dataset"

# Liste de tous les fichiers .h5 dans le dossier
h5_files = glob.glob(os.path.join(dataset_path, "*.h5"))

# Listes pour stocker les DataFrames normaux et anormaux
normal_data_frames = []
anomaly_data_frames = []

# Charger chaque fichier .h5 dans un DataFrame et le classer
for file in h5_files:
    try:
        df = pd.read_hdf(file)
        
        # Identifier les fichiers anormaux par la présence de mots-clés dans le nom du fichier
        if "shift_on" in file:
            anomaly_data_frames.append(df)
            print(f"{file} classé comme anomalie.")
        else:
            normal_data_frames.append(df)
            print(f"{file} classé comme normal.")
            
    except Exception as e:
        print(f"Erreur lors du chargement de {file}: {e}")




# Uniformiser toutes les tailles des DataFrames à (3581, 135)
for i in range(len(normal_data_frames)):
    if normal_data_frames[i].shape[0] > 3581:
        # Supprimer les lignes excédentaires à partir de la fin
        normal_data_frames[i] = normal_data_frames[i].iloc[:3581, :]

for i in range(len(anomaly_data_frames)):
    if anomaly_data_frames[i].shape[0] > 3581:
        # Supprimer les lignes excédentaires à partir de la fin
        anomaly_data_frames[i] = anomaly_data_frames[i].iloc[:3581, :]


        
# Colonnes à supprimer
columns_to_drop = [
    'sim/cockpit/autopilot/flight_director_roll_ground_truth',
    'sim/cockpit/autopilot/flight_director_pitch_ground_truth'
]

# Suppression des colonnes des DataFrames dans anomaly_data_frames
for i in range(len(anomaly_data_frames)):
    anomaly_data_frames[i] = anomaly_data_frames[i].drop(columns=columns_to_drop, errors='ignore')




# Diviser les données normales en ensemble d'entraînement (80%) et ensemble de test (20%)
train_normal, test_normal = train_test_split(normal_data_frames, test_size=0.2, random_state=42)

# Les données anormales seront utilisées comme ensemble de test
test_anomaly = anomaly_data_frames


# Fonction pour normaliser et convertir les DataFrames en tableau numpy
def prepare_data(dataframes):
    """
    Prépare les données pour le modèle LSTM:
    - Convertit les DataFrames en tableaux numpy.
    - Normalise les valeurs avec StandardScaler.

    Paramètres:
    - dataframes (list): Liste de DataFrames.

    Retourne:
    - numpy.array: Données normalisées sous forme de tableau numpy.
    """
    scaler = StandardScaler()
    # Convertir les DataFrames en tableau numpy et normaliser
    data_array = [df.values for df in dataframes]
    data_array = np.array([scaler.fit_transform(simulation) for simulation in data_array])
    return np.array(data_array)

# Normalisation et conversion des données
train_normal_data = prepare_data(train_normal)
test_normal_data = prepare_data(test_normal)
test_anomaly_data = prepare_data(test_anomaly)



# Définir les paramètres des données
timesteps = 3581  # Nombre de pas temporels
features = 135    # Nombre de caractéristiques

inputs = tf.keras.layers.Input(shape=(timesteps, features))

# LSTM Layers
encoded = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)(inputs)
encoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False)(encoded)

# Bottleneck
bottleneck = tf.keras.layers.Dense(32, activation='relu')(encoded)

# Decoder
decoded = tf.keras.layers.RepeatVector(3581)(bottleneck)
decoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(decoded)
decoded = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)(decoded)
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(135))(decoded)


autoencoder = tf.keras.Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

from tensorflow.keras.callbacks import ModelCheckpoint

# Définir le callback pour sauvegarder le modèle complet
checkpoint_callback = ModelCheckpoint(
    filepath='autoencoder_model.keras',  # Sauvegarde du modèle complet
    save_weights_only=False,             # Sauvegarder le modèle entier
    save_best_only=True,                 # Sauvegarder uniquement les meilleurs modèles
    monitor='val_loss',                  # Suivre 'val_loss'
    mode='min',                          # Sauvegarder si 'val_loss' diminue
    verbose=1                            # Afficher les informations de sauvegarde
)

# Entraînement avec sauvegarde
with tf.device('/GPU:0'):
    history = autoencoder.fit(
        train_normal_data, train_normal_data,
        epochs=5,
        batch_size=16,
        validation_data=(test_normal_data, test_normal_data),
        shuffle=True,
        callbacks=[checkpoint_callback]  # Ajouter le callback ici
    )

import matplotlib.pyplot as plt

# Tracer la courbe de la loss
def plot_and_save_loss(history, filename='loss_plot.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Enregistrer le graphe
    plt.show()

# Après l'entraînement
plot_and_save_loss(history)

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


#réduire le nbre de timesteps 


def reduce_timesteps(data, new_timesteps):
    """
    Réduit le nombre de timesteps des données de manière stricte.

    Args:
        data (numpy array): Tableau de forme (samples, timesteps, features).
        new_timesteps (int): Nombre de timesteps désirés.

    Returns:
        numpy array: Données ajustées pour correspondre exactement à new_timesteps.
    """
    original_timesteps = data.shape[1]
    if new_timesteps > original_timesteps:
        raise ValueError("Le nouveau nombre de timesteps dépasse la taille originale.")
    
    # Calculer le nombre de pas à conserver
    start = (original_timesteps - new_timesteps) // 2
    end = start + new_timesteps

    # Découper les séquences
    reduced_data = data[:, start:end, :]
    return reduced_data


new_timesteps = 1000
train_normal_data_reduced = reduce_timesteps(train_normal_data, new_timesteps)
test_normal_data_reduced = reduce_timesteps(test_normal_data, new_timesteps)
test_anomaly_data_reduced = reduce_timesteps(test_anomaly_data, new_timesteps)


#import zipfile

# Chemin vers le fichier ZIP
#zip_path = "autoencoder_models.keras"

# Extraire les fichiers
#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#    zip_ref.extractall("extracted_model")



# Définir les paramètres des données
timesteps = 3581  # Nombre de pas temporels
features = 135    # Nombre de caractéristiques

inputs = tf.keras.layers.Input(shape=(new_timesteps, features))


# LSTM Layers
encoded = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)(inputs)
encoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False)(encoded)

# Bottleneck
bottleneck = tf.keras.layers.Dense(32, activation='relu')(encoded)

# Decoder
decoded = tf.keras.layers.RepeatVector(new_timesteps)(bottleneck)
decoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(decoded)
decoded = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)(decoded)
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(135))(decoded)


autoencoder = tf.keras.Model(inputs, outputs)


import zipfile
import os

# Chemin vers le fichier ZIP contenant le modèle
zip_path = os.path.join(os.path.dirname(__file__), "autoencoder_model.keras")

# Dossier temporaire pour extraire les fichiers
extraction_dir = os.path.join(os.path.dirname(__file__), "extracted_model")

# Extraire les fichiers si nécessaire
if not os.path.exists(extraction_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
        print(f"Fichiers extraits dans : {extraction_dir}")

# Chemin des poids directement dans extracted_model
weights_path = os.path.join(extraction_dir, "model.weights.h5")

# Vérification de l'existence des poids
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Le fichier de poids est introuvable : {weights_path}")

# Charger les poids dans le modèle
autoencoder.load_weights(weights_path)
print(f"Poids chargés depuis : {weights_path}")

# Étape 1 : Filtrer les colonnes des données originales (en excluant ground truth)
columns_to_keep = [col for col in test_anomaly[0].columns if "ground_truth" not in col]

# Recréer les données originales filtrées
test_anomaly_filtered = [
    df[columns_to_keep] for df in test_anomaly
]

# Préparer les données filtrées
test_anomaly_data_filtered = prepare_data(test_anomaly_filtered)

# Réduire les timesteps
test_anomaly_data_filtered_reduced = reduce_timesteps(test_anomaly_data_filtered, new_timesteps)

# Reconstruction avec le modèle
reconstructed_data_test = autoencoder.predict(test_anomaly_data_filtered_reduced)

# Étape 2 : Calcul de l'erreur de reconstruction
reconstruction_errors_test = np.mean(
    np.square(test_anomaly_data_filtered_reduced - reconstructed_data_test),
    axis=(1, 2)
)

# Reconstruire les séries temporelles
reconstructed_data_train = autoencoder.predict(train_normal_data_reduced)
reconstruction_errors_train = np.mean(np.square(train_normal_data_reduced - reconstructed_data_train), axis=(1, 2))


# Détecter les anomalies
threshold = np.percentile(reconstruction_errors_train, 95)
anomalies = reconstruction_errors_test > threshold
print(f"Nombre d'anomalies détectées : {np.sum(anomalies)}")

# Colonnes d'intérêt
roll_predicted = "sim/cockpit/autopilot/flight_director_roll"
pitch_predicted = "sim/cockpit/autopilot/flight_director_pitch"
roll_ground_truth = "sim/cockpit/autopilot/flight_director_roll_ground_truth"
pitch_ground_truth = "sim/cockpit/autopilot/flight_director_pitch_ground_truth"

# Extraire les colonnes originales et reconstruites
def extract_columns(original_data, reconstructed_data, column_index):
    """
    Extrait une colonne spécifique des données originales et reconstruites.
    Args:
        original_data (numpy array): Données originales.
        reconstructed_data (numpy array): Données reconstruites.
        column_index (int): Index de la colonne d'intérêt.
    Returns:
        tuple: (valeurs originales, valeurs reconstruites)
    """
    original = original_data[:, :, column_index]
    reconstructed = reconstructed_data[:, :, column_index]
    return original, reconstructed

# Indices des colonnes dans les DataFrames
roll_index = test_anomaly[0].columns.get_loc(roll_predicted)
pitch_index = test_anomaly[0].columns.get_loc(pitch_predicted)
roll_truth_index = test_anomaly[0].columns.get_loc(roll_ground_truth)
pitch_truth_index = test_anomaly[0].columns.get_loc(pitch_ground_truth)

# Extraire les valeurs pour les données de test
roll_original, roll_reconstructed = extract_columns(test_anomaly_data_reduced, reconstructed_data_test, roll_index)
pitch_original, pitch_reconstructed = extract_columns(test_anomaly_data_reduced, reconstructed_data_test, pitch_index)
roll_truth = test_anomaly_data_reduced[:, :, roll_truth_index]
pitch_truth = test_anomaly_data_reduced[:, :, pitch_truth_index]

# Fonction d'affichage
import matplotlib.pyplot as plt

def plot_comparison(original, reconstructed, truth, title, filename):
    """
    Affiche la comparaison entre les données originales, reconstruites et la vérité terrain.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original[0], label="Original", color='blue')
    plt.plot(reconstructed[0], label="Reconstruit", linestyle="--", color='orange')
    plt.plot(truth[0], label="Vérité terrain", linestyle=":", color='green')
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Comparaison des reconstructions
plot_comparison(roll_original, roll_reconstructed, roll_truth, 
                "Comparaison de flight_director_roll (Test Anomalies)", "roll_comparison_test.png")

plot_comparison(pitch_original, pitch_reconstructed, pitch_truth, 
                "Comparaison de flight_director_pitch (Test Anomalies)", "pitch_comparison_test.png")


import matplotlib.pyplot as plt
import numpy as np

def plot_anomalies(reconstruction_errors, threshold, filename="anomaly_detection.png"):
    """
    Génère un graphique montrant les erreurs de reconstruction et identifie les anomalies.

    Args:
        reconstruction_errors (numpy array): Erreurs de reconstruction pour chaque échantillon.
        threshold (float): Seuil au-delà duquel une erreur est considérée comme une anomalie.
        filename (str): Nom du fichier pour sauvegarder le graphique.
    """
    plt.figure(figsize=(12, 6))
    
    # Tracer les erreurs de reconstruction
    plt.plot(reconstruction_errors, label="Erreur de reconstruction", color='blue')
    
    # Tracer la ligne de seuil
    plt.axhline(y=threshold, color='red', linestyle='--', label=f"Seuil ({threshold:.2f})")
    
    # Ajouter des annotations
    plt.title("Détection des anomalies basée sur l'erreur de reconstruction")
    plt.xlabel("Index de l'échantillon")
    plt.ylabel("Erreur de reconstruction")
    plt.legend()
    plt.grid(True)
    
    # Sauvegarder le graphique
    plt.savefig(filename)
    plt.show()

# Exemple d'utilisation
# reconstruction_errors = np.mean(np.square(test_normal_data_reduced - reconstructed_data), axis=(1, 2))
# threshold = np.percentile(reconstruction_errors, 95)
plot_anomalies(reconstruction_errors_test, threshold, filename="anomaly_detection.png")



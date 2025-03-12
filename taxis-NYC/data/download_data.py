import pandas as pd
from sklearn.model_selection import train_test_split
import sqlite3
import zipfile
import requests
import os
import shutil

import common

# Configuration des chemins
DB_PATH = os.path.join( 'data','taxis.db')  # Chemin de la base de données
ZIP_URL = common.CONFIG['paths']['zip_url']  # URL du fichier ZIP
ZIP_PATH = common.CONFIG['paths']['zip_path']  # Chemin local du fichier ZIP
EXTRACT_FOLDER = common.CONFIG['paths']['extract_folder']  # Dossier d'extraction
RANDOM_STATE = int(common.CONFIG['ml']['random_state'])  # Seed pour la reproductibilité

def extract_data():
    """
    Télécharge et extrait les données des taxis de New York depuis une URL.
    Retourne le chemin du fichier CSV extrait.
    """
    # Vérifie si le dossier data existe
    if not os.path.exists(os.path.dirname(ZIP_PATH)):
        os.makedirs(os.path.dirname(ZIP_PATH))

    # Télécharger le fichier ZIP
    print("Téléchargement des données...")
    try:
        response = requests.get(ZIP_URL, stream=True)
        response.raise_for_status()  # Lève une exception si le téléchargement échoue
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        raise

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Téléchargement terminé.")

    # Extraction du fichier ZIP
    print("Extraction du fichier ZIP...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
        print(f"✅ Fichiers extraits dans {EXTRACT_FOLDER}")
    except zipfile.BadZipFile:
        print("❌ Le fichier ZIP est corrompu.")
        raise
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction : {e}")
        raise

    # Recherche d'un fichier CSV dans le dossier extrait
    csv_files = [f for f in os.listdir(EXTRACT_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("❌ Aucun fichier CSV trouvé après extraction.")

    csv_path = os.path.join(EXTRACT_FOLDER, csv_files[0])
    print(f"✅ Chargement des données depuis {csv_path}")
    return csv_path

def download_data():
    """
    Télécharge, extrait et charge les données dans une base de données SQLite.
    """
    csv_path = extract_data()

    # Charger le fichier CSV dans Pandas
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier CSV : {e}")
        raise

    # Séparer les données en train/test
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

    # Sauvegarde en base de données
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    print(f"Enregistrement des données dans la base SQLite : {DB_PATH}")
    try:
        with sqlite3.connect(DB_PATH) as con:
            data_train.to_sql(name='train', con=con, if_exists="replace", index=False)
            data_test.to_sql(name='test', con=con, if_exists="replace", index=False)
        print("✅ Données sauvegardées avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement en base de données : {e}")
        raise

    # Suppression des fichiers temporaires
    print("🧹 Nettoyage des fichiers temporaires...")
    try:
        os.remove(ZIP_PATH)  # Supprime le fichier ZIP
        shutil.rmtree(EXTRACT_FOLDER, ignore_errors=True)  # Supprime le dossier d'extraction
        print("✅ Fichiers temporaires supprimés.")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression des fichiers temporaires : {e}")

def test_download_data():
    """
    Teste la présence des données dans la base de données SQLite.
    """
    print(f"Lecture des données depuis la base de données : {DB_PATH}")
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()

            # Données d'entraînement
            res = cur.execute("SELECT COUNT(*) FROM train")
            n_rows = res.fetchone()[0]
            res = cur.execute("SELECT * FROM train LIMIT 1")
            n_cols = len(res.description)
            print(f'Données d\'entraînement : {n_rows} lignes x {n_cols} colonnes')

            # Données de test
            res = cur.execute("SELECT COUNT(*) FROM test")
            n_rows = res.fetchone()[0]
            res = cur.execute("SELECT * FROM test LIMIT 1")
            n_cols = len(res.description)
            print(f'Données de test : {n_rows} lignes x {n_cols} colonnes')
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de la base de données : {e}")
        raise

if __name__ == "__main__":
    try:
        download_data()
        test_download_data()
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")
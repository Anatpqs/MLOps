import pandas as pd
from google.cloud import storage
import os


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Charge un seul fichier CSV depuis un bucket Google Cloud Storage.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        os.getcwd(), "service_account.json"
    )
    storage_client = storage.Client(project=project)
    bucket_name = bucket_path.split("/")[0]
    folder = "/".join(bucket_path.split("/")[1:])

    # Liste les fichiers dans le bucket
    blobs = list(storage_client.list_blobs(bucket_name, prefix=folder))

    if not blobs:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans le bucket {bucket_name} avec le préfixe {folder}"
        )

    # Prend le premier fichier trouvé
    blob = blobs[0]

    # Télécharge le fichier temporairement
    temp_dir = os.path.join(os.getcwd(), "tmp")
    temp_path = os.path.join(temp_dir, blob.name.split("/")[-1])
    blob.download_to_filename(temp_path)

    # Charge le fichier en DataFrame
    df = pd.read_csv(temp_path)

    # Supprime le fichier temporaire après chargement
    os.remove(temp_path)

    return df

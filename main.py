from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict

# Chemins vers les fichiers
DATA_PATH = './small_data.csv'
CLIENT_INFO_PATH = './appli_train_small.csv'  # Nouveau fichier pour les informations originelles des clients liés à small_data
SCALER_PATH = './scaler.joblib'
MODEL_PATH = './model.joblib'

# Fonctions pour charger les données, le scaler et le modèle
def load_data():
    return pd.read_csv(DATA_PATH)

def load_client_info():
    return pd.read_csv(CLIENT_INFO_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

# Charge les données
df = load_data()
client_info_df = load_client_info()
scaler = load_scaler()
model = load_model()

# FastAPI app
app = FastAPI()

# Fonction pour récupérer les données d'un client
def get_client_data(client_id: int):
    client_row = df[df['SK_ID_CURR'] == client_id]
    if client_row.empty:
        return None  # Client ID non trouvé
    return client_row

# Fonction pour récupérer les informations descriptives d'un client
def get_client_info(client_id: int):
    client_info = client_info_df[client_info_df['SK_ID_CURR'] == client_id]
    if client_info.empty:
        return None  # Client ID non trouvé
    return client_info.to_dict(orient="records")[0]

# Fonction pour récupérer l'importance des features globales
def get_global_feature_importance():
    return model.feature_importances_

# Fonction pour récupérer l'importance des features locales
def get_local_feature_importance(client_id: int):
    client_features = get_client_data(client_id)
    if client_features is None:
        return None
    client_features_2d = np.array(client_features).reshape(1, -1)
    scaled_features = scaler.transform(client_features_2d)
    return model.feature_importances_ * scaled_features

@app.get("/predict/{client_id}")
def predict(client_id: int):
    client_features = get_client_data(client_id)
    if client_features is None:
        raise HTTPException(status_code=404, detail="Client not found")
    client_features_2d = np.array(client_features).reshape(1, -1)
    scaled_features = scaler.transform(client_features_2d)
    proba = model.predict_proba(scaled_features)
    proba = proba[:, 1]
    seuil = 0.07
    classe = "Accepté" if proba <= seuil else "Refusé"
    return {"client_id": client_id, "probabilité": proba[0], "classe": classe}

@app.get("/client_info/{client_id}")
def client_info(client_id: int):
    client_info = get_client_info(client_id)
    if client_info is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return client_info

@app.get("/feature_importance/global")
def feature_importance_global():
    return {"feature_importance": get_global_feature_importance().tolist()}

@app.get("/feature_importance/local/{client_id}")
def feature_importance_local(client_id: int):
    local_importance = get_local_feature_importance(client_id)
    if local_importance is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"feature_importance": local_importance.tolist()}

##MAJ des données clients

class ClientUpdate(BaseModel):
    features: Dict[str, float]

@app.put("/update_client/{client_id}")
def update_client(client_id: int, update: ClientUpdate):
    global df, client_info_df
    client_row = df[df['SK_ID_CURR'] == client_id]
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    for feature, value in update.features.items():
        df.loc[df['SK_ID_CURR'] == client_id, feature] = value
        if feature in client_info_df.columns:
            client_info_df.loc[client_info_df['SK_ID_CURR'] == client_id, feature] = value
    df.to_csv(DATA_PATH, index=False)
    client_info_df.to_csv(CLIENT_INFO_PATH, index=False)
    return {"message": "Client updated successfully"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
#####
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import os
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.api.types import CategoricalDtype

# URL de base de l'API FastAPI
#BASE_URL = "https://limitless-garden-12149-a788ad4be814.herokuapp.com"
BASE_URL = "https://arcane-dusk-90161-219ca85f1f94.herokuapp.com"
#BASE_URL = "http://127.0.0.1:8000"

CLIENT_INFO_PATH = './appli_train_small.csv'
#DATA_PATH = './small_data.csv'



# Colonnes à afficher dans Streamlit
colonnes_2keep = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY','OCCUPATION_TYPE']

def load_client_info():
    try:
        if not os.path.exists(CLIENT_INFO_PATH):
            raise FileNotFoundError(f"Fichier introuvable : {CLIENT_INFO_PATH}")
        return pd.read_csv(CLIENT_INFO_PATH, usecols=colonnes_2keep)
    except Exception as e:
        print(f"[ERREUR] Échec du chargement de client_info_df : {e}")
        return pd.DataFrame(columns=colonnes_2keep)

df = load_client_info()

def prettify(name):
    """Nettoie les noms de colonnes pour les titres"""
    return name.replace('_', ' ').title()
#=======================================================================================================================================
# fonction de traitement graphique comparaison client
def plot_feature_distribution(df, feature, client_value, client_id="Client"):
    if is_numeric_dtype(df[feature]):
        # Préparer les données triées
        sorted_df = df[[feature]].dropna().copy()
        sorted_df = sorted_df.sort_values(by=feature).reset_index(drop=True)
        sorted_df["ClientIndex"] = sorted_df.index.astype(str)

        fig = px.bar(sorted_df, x=feature, y="ClientIndex", orientation="h",
                     title=f"Valeur de {feature} pour chaque client")

        # Position la plus proche du client
        closest_idx = (sorted_df[feature] - client_value).abs().idxmin()
        fig.add_trace(go.Scatter(
            x=[client_value], y=[str(closest_idx)],
            mode="markers",
            marker=dict(size=14, color='red', symbol='circle'),
            name=f"{client_id}"
        ))

        subtitle = f"<span style='font-size:16px;color:red'><b>{client_id} – Valeur : {client_value}</b></span>"

    elif df[feature].nunique() < 20:
        # Traitement catégoriel
        counts = df[feature].astype(str).value_counts().reset_index()
        counts.columns = [feature, 'count']
        counts['pull'] = counts[feature].apply(lambda x: 0.1 if x == str(client_value) else 0)

        fig = px.pie(counts, names=feature, values='count',
                     title=f"Répartition de {feature}",
                     hole=0.3)

        fig.update_traces(
            pull=counts['pull'],  
            textinfo='label+percent',
            marker=dict(line=dict(color='black', width=1))
        )

        subtitle = f"<span style='font-size:16px;color:red'><b>{client_id} – Catégorie : {client_value}</b></span>"

    else:
        st.warning(f"Type de variable non pris en charge pour {feature}.")
        return

    # Mise en forme WCAG-friendly
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', size=14),
        title=dict(
            text=f"Répartition de {feature}<br><span style='font-size:16px;color:red;'>Valeur client : {client_value}</span>",
            font=dict(size=18, color='black'),
            x=0.5),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(color="black")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

#=======================================================================================================================================
#FONCTION D'ANALYSE BIVAIREE

def plot_bivariate_analysis(df,feature_x,feature_y, client_value_x=None, client_value_y=None, client_id="Client"):
   
        x = df[feature_x]
        y = df[feature_y]



        # Détecter les types
        x_is_num = is_numeric_dtype(x)
        y_is_num = is_numeric_dtype(y)
        x_is_cat = is_string_dtype(x) or isinstance(x.dtype, CategoricalDtype)
        y_is_cat = is_string_dtype(y) or isinstance(y.dtype, CategoricalDtype)

        fig = None

        if x_is_num and y_is_cat:
            title = f"Boxplot de {prettify(feature_x)} selon {prettify(feature_y)}"
            fig = px.box(df, x=feature_y, y=feature_x, points="all", title=title)
                # Corriger titres axes
            x_title = prettify(feature_y)
            y_title = prettify(feature_x)

        elif x_is_cat and y_is_num:
            title = f"Boxplot de {prettify(feature_y)} selon {prettify(feature_x)}"
            fig = px.box(df, x=feature_x, y=feature_y, points="all", title=title)
            x_title = prettify(feature_x)
            y_title = prettify(feature_y)

        elif x_is_cat and y_is_cat:
            title = f"Heatmap : {prettify(feature_x)} vs {prettify(feature_y)}"
            ct = pd.crosstab(df[feature_x], df[feature_y])
            fig = px.imshow(ct, text_auto=True, title=title)
            x_title = prettify(feature_y)
            y_title = prettify(feature_x)

        elif x_is_num and y_is_num:
            title = f"Nuage de points : {prettify(feature_x)} vs {prettify(feature_y)}"
            fig = px.scatter(df, x=feature_x, y=feature_y, trendline="ols", title=title)
            x_title = prettify(feature_x)
            y_title = prettify(feature_y)

            # Positionnement du client
            if client_value_x is not None and client_value_y is not None:
                fig.add_trace(go.Scatter(
                    x=[client_value_x],
                    y=[client_value_y],
                    mode="markers",
                    marker=dict(size=14, color='red', symbol='circle'),
                    name=f"{client_id}"
                ))             
        else:
            st.warning("Type de variables non pris en charge.")

        if fig:
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,  # Centré
                    xanchor='center',
                    font=dict(size=20, color='black')
                ),
                font=dict(color='black'),  # Texte général
                paper_bgcolor='white',
                plot_bgcolor='white',
                legend=dict(
                    font=dict(color='black'),
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    orientation='v',
                    x=1.02,  # Légende à droite
                    xanchor='left'
                ),
                xaxis=dict(
                    title=x_title,
                    title_font=dict(color='black'),
                    tickfont=dict(color='black'),
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                yaxis=dict(
                    title=y_title,
                    title_font=dict(color='black'),
                    tickfont=dict(color='black'),
                    linecolor='black',
                    gridcolor='lightgrey'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

def generate_description_bivarie(df,feature_x,feature_y):
    x = df[feature_x]
    y = df[feature_y]

    # Détecter les types
    x_is_num = is_numeric_dtype(x)
    y_is_num = is_numeric_dtype(y)
    x_is_cat = is_string_dtype(x) or isinstance(x.dtype, CategoricalDtype)
    y_is_cat = is_string_dtype(y) or isinstance(y.dtype, CategoricalDtype)
    
    if x_is_num and y_is_cat:
        description = f"""
        **Analyse  de {prettify(feature_x)} en fonction de {prettify(feature_y)}" :**
        - Le graphique représente pour chaque catégorie de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_y)}</span> la répartition des valeurs de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_x)}</span> sous forme de nuage de points.
        - A côté, pour chaque catégorie, nous avons une boite à moustache représentant les 4 quartiles de valeurs ainsi que la médiane.
        """
    if x_is_cat and y_is_num:
        description = f"""
        **Analyse  de {prettify(feature_y)} en fonction de {prettify(feature_x)}" :**
        - Le graphique représente pour chaque catégorie <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_x)}</span> la répartition des valeurs de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_y)}</span> sous forme de nuage de points.
        - A côté, pour chaque catégorie, nous avons une boite à moustache représentant les 4 quartiles de valeurs ainsi que la médiane.
        """
    if x_is_cat and y_is_cat:
        description = f"""
        **Analyse  de {prettify(feature_x)} en fonction de {prettify(feature_y)}" :**
        - Le graphique est une heatmap qui représente la répartition des population de chaque catégorie de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_x)}</span> dans chacune des catégories de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_y)}</span>.
        - La couleur de chaque case varie du bleu foncé au rouge selon l'importance de la valeur
        """
    if x_is_num and y_is_num:
        description = f"""
        **Analyse  de {prettify(feature_x)} en fonction de {prettify(feature_y)}" :**
        - Le graphique est un nuage de points qui positionne les clients en fonction de la valeur de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_x)}</span> sur l'axe horizontal la valeurs de <span style='font-size:20px; text-decoration:underline;'>{prettify(feature_y)}</span>.
        - Une droite représente une possible corrélation entre les deux valeurs sur toute la population.
        - Un point rouge indique où le client sélectionné est positionné.
        """

    return description

#=====================================================CODE STREAMLIT=========================================================================

st.set_page_config(
    page_title="Dashboard de scoring Crédit Client",
    layout="wide",  # ✅ Ce paramètre utilise toute la largeur de la page
    initial_sidebar_state="auto"
)
if not df.empty:
    st.title("Dashboard de scoring Crédit Client")

    # Entrée utilisateur
    client_id = st.number_input("Entrez le Client ID", min_value=0, step=1)

    # Division en deux colonnes principales
    col_gauche, col_droite = st.columns([1, 2])

    # === COLONNE GAUCHE ===
    with col_gauche:
        st.subheader("Prédiction et Informations Client")

    # Bouton pour déclencher la prédiction
        if st.button("Prédire"):
            response = requests.get(f"{BASE_URL}/predict/{client_id}")
            if response.status_code == 200:
                result = response.json()
                st.markdown(f"**Client ID** : {result['client_id']}")
                st.markdown(f"**Probabilité de défaut** : {result['probabilité']:.4f}")
                st.markdown(f"**Classe prédite** : {result['classe']}")

                # Jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['probabilité'],
                    title={'text': "Score de Crédit"},
                    gauge={'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.07], 'color': "lightgreen"},
                            {'range': [0.07, 1], 'color': "red"}
                        ]}
                ))
                st.plotly_chart(fig)

                # Infos client
                response_info = requests.get(f"{BASE_URL}/client_info/{client_id}")
                if response_info.status_code == 200:
                    client_info = response_info.json()
                    filtered_info = {k: v for k, v in client_info.items() if k in colonnes_2keep}
                    st.subheader("Informations Client")
                    st.json(filtered_info)
                else:
                    st.error("Erreur lors de la récupération des informations client.")
            elif response.status_code == 404:
                st.error("Client non trouvé")
            else:
                st.error(f"Erreur: {response.status_code}")

    # === COLONNE DROITE ===
    with col_droite:

    # Importance des features
        
        st.subheader("Importance des Features")
        if st.button("Visualiser"): 
            global_response = requests.get(f"{BASE_URL}/feature_importance/global")
            local_response = requests.get(f"{BASE_URL}/feature_importance/local/{client_id}")

            if global_response.status_code == 200 and local_response.status_code == 200:
                global_img = Image.open(BytesIO(global_response.content))
                local_img = Image.open(BytesIO(local_response.content))

                # Affichage côte à côte avec Streamlit
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Globale**")
                    st.image(global_img, use_container_width=True)
                with col2:
                    st.markdown("**Locale**")
                    st.image(local_img, use_container_width=True)
            else:
                st.warning("Erreur lors de la récupération des graphiques d'importance.")

            st.markdown("---")

    #Comparaison des caractéristiques des clients   
        st.subheader("Comparaison des Caractéristiques")
        options = ['Sélectionnez une feature'] + list(df.columns[1:])
        feature = st.selectbox("Sélectionnez une feature", options, key="comp_feature")
        # Gérer le cas où aucune vraie feature n’est sélectionnée
        if feature == 'Sélectionnez une feature':
            st.warning("Veuillez sélectionner deux features valides.")
        else:
            st.write(f"Feature sélectionnée : {feature}")

        if st.button("Comparer"):
            response = requests.get(f"{BASE_URL}/client_info/{client_id}")
            if response.status_code == 200:
                client_info = response.json()
                client_value = client_info[feature]
                plot_feature_distribution(df, feature, client_value, client_id=client_id)
            else:
                st.error("Erreur lors de la récupération des données client.")

        st.markdown("---")
    # Analyse bivariée    
        st.subheader("Analyse Bivariée")

        #creation des box pour la sélection des features
        response = requests.get(f"{BASE_URL}/client_info/{client_id}")
        options = ['Sélectionnez une feature'] + list(df.columns[1:])

        feature_1 = st.selectbox("Sélectionnez la première feature", options, key="biv1")
        feature_2 = st.selectbox("Sélectionnez la deuxième feature", options, key="biv2")

        # Gérer le cas où aucune vraie feature n’est sélectionnée
        if feature_1 == 'Sélectionnez une feature' or feature_2 == 'Sélectionnez une feature':
            st.warning("Veuillez sélectionner deux features valides.")
        else:
            st.write(f"Feature 1 sélectionnée : {feature_1}")
            st.write(f"Feature 2 sélectionnée : {feature_2}")

        if response.status_code == 200:
            client_info = response.json()
            client_value1 = client_info[feature_1]
            client_value2 = client_info[feature_2]
        else:
            st.error("Erreur lors de la récupération des données client.")
        if st.button("Analyser"):
            plot_bivariate_analysis(df, feature_1, feature_2, client_value_x=client_value1, client_value_y=client_value2, client_id=client_id)
            describe = generate_description_bivarie(df,feature_1,feature_2)
            st.markdown(describe, unsafe_allow_html=True)
            st.download_button(
                label="Télécharger la description du graphique",
                data=describe,
                file_name="description_graphique_bivarie.txt",
                mime="text/plain"
        )
else:
    st.warning("Données clients indisponible")

# =========================================================
# Prévision simple et robuste de la consommation électrique
# Sénégal – Dakar / Régions
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import datetime
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Prévision Consommation Électrique Sénégal",
    layout="wide"
)

st.title("⚡ Prévision de la consommation électrique – Sénégal")
st.write(
    "Application de prévision **robuste et réaliste** de la consommation "
    "électrique horaire, basée sur un modèle d’apprentissage automatique "
    "sans ajustements artificiels."
)

# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================
@st.cache_data
def load_data():
    base_path = os.path.join(os.path.dirname(__file__), "data")
    dakar = pd.read_csv(
        os.path.join(base_path, "dakar_new.csv"),
        parse_dates=["Datetime", "Date"]
    )
    region = pd.read_csv(
        os.path.join(base_path, "region_new.csv"),
        parse_dates=["Datetime", "Date"]
    )
    return dakar, region

dakar, region = load_data()

# =========================================================
# OUTILS
# =========================================================
def extract_hour(col):
    if col.dtype == "O":
        return pd.to_datetime(col, errors="coerce").dt.hour
    return col.astype(int)

jours_feries = pd.to_datetime([
    "2024-01-01", "2024-04-04", "2024-04-10", "2024-05-01",
    "2024-06-16", "2024-07-16", "2024-08-15", "2024-12-25",
    "2025-01-01", "2025-04-04", "2025-05-01", "2025-06-09",
    "2025-07-05", "2025-08-13", "2025-12-25"
])

ramadan_dates = {
    2024: pd.date_range("2024-03-11", "2024-04-09"),
    2025: pd.date_range("2025-03-01", "2025-03-30"),
    2026: pd.date_range("2026-02-18", "2026-03-19")
}

def is_ramadan(date):
    date = pd.to_datetime(date)
    return int(date.year in ramadan_dates and date in ramadan_dates[date.year])

def is_ferie(date):
    return int(pd.to_datetime(date) in jours_feries)

# =========================================================
# MODÈLE DE PRÉVISION SIMPLIFIÉ ET SÉCURISÉ
# =========================================================
def pred_base_simple(
    df_base,
    colonne_cons,
    date_choisie,
    temp_hour,
    humid_hour
):
    df = df_base.copy()

    # ----------------------------
    # Sécurisation météo
    # ----------------------------
    if "temperature" not in df.columns:
        df["temperature"] = 28.0   # valeur moyenne Sénégal

    if "humidity" not in df.columns:
        df["humidity"] = 60.0      # valeur neutre

    # ----------------------------
    # Feature engineering
    # ----------------------------
    df["hour"] = extract_hour(df["Heure"])
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_ramadan"] = df["Date"].apply(is_ramadan)
    df["is_ferie"] = df["Date"].apply(is_ferie)

    features = [
        "hour",
        "month",
        "day_of_week",
        "is_weekend",
        "temperature",
        "humidity",
        "is_ramadan",
        "is_ferie"
    ]

    # Nettoyage
    df = df.dropna(subset=features + [colonne_cons])

    X = df[features]
    y = df[colonne_cons]

    # ----------------------------
    # Split temporel
    # ----------------------------
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ----------------------------
    # Modèle
    # ----------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ----------------------------
    # Évaluation
    # ----------------------------
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)

    # ----------------------------
    # Jour à prédire
    # ----------------------------
    date_dt = pd.to_datetime(date_choisie)

    df_pred = pd.DataFrame({
        "hour": np.arange(24),
        "temperature": temp_hour,
        "humidity": humid_hour
    })

    df_pred["month"] = date_dt.month
    df_pred["day_of_week"] = date_dt.weekday()
    df_pred["is_weekend"] = int(date_dt.weekday() >= 5)
    df_pred["is_ramadan"] = is_ramadan(date_dt)
    df_pred["is_ferie"] = is_ferie(date_dt)

    # ----------------------------
    # Prédiction
    # ----------------------------
    y_pred = model.predict(df_pred[features])

    # ----------------------------
    # Graphique
    # ----------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pred["hour"],
        y=y_pred,
        mode="lines+markers",
        name="Consommation prédite"
    ))

    fig.update_layout(
        title=f"{colonne_cons} – {date_choisie} ",
        xaxis_title="Heure",
        yaxis_title="Consommation (MW)",
        template="plotly_white"
    )

    return y_pred, fig, mae

# =========================================================
# INTERFACE UTILISATEUR
# =========================================================
date_choisie = st.date_input(
    "📅 Choisissez une date :",
    datetime.date(2025, 10, 13)
)

st.subheader("🌡️ Saisie horaire température et humidité")

temp_hour = []
humid_hour = []

cols = st.columns(2)
for h in range(24):
    with cols[0]:
        temp_hour.append(
            st.number_input(
                f"T° heure {h}",
                min_value=0.0,
                max_value=45.0,
                value=28.0,
                key=f"temp_{h}"
            )
        )
    with cols[1]:
        humid_hour.append(
            st.number_input(
                f"Humidité heure {h} (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                key=f"hum_{h}"
            )
        )

# =========================================================
# LANCEMENT PRÉVISION
# =========================================================
if st.button("🚀 Lancer la prévision"):
    with st.spinner("Calcul des prévisions..."):

        y_dakar, fig_dakar, mae_dakar = pred_base_simple(
            dakar,
            "cons_dakar",
            date_choisie,
            temp_hour,
            humid_hour
        )

        y_region, fig_region, mae_region = pred_base_simple(
            region,
            "cons_regions",
            date_choisie,
            temp_hour,
            humid_hour
        )

        heures = np.arange(24)
        df_res = pd.DataFrame({
            "Heure": heures,
            "Dakar (MW)": y_dakar,
            "Régions (MW)": y_region
        })
        df_res["Total Sénégal (MW)"] = (
            df_res["Dakar (MW)"] + df_res["Régions (MW)"]
        )

        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(x=heures, y=df_res["Dakar (MW)"], name="Dakar"))
        fig_total.add_trace(go.Scatter(x=heures, y=df_res["Régions (MW)"], name="Régions"))
        fig_total.add_trace(go.Scatter(
            x=heures,
            y=df_res["Total Sénégal (MW)"],
            name="Total Sénégal",
            line=dict(width=3)
        ))

        fig_total.update_layout(
            title=f"Prévision totale – {date_choisie}",
            xaxis_title="Heure",
            yaxis_title="Consommation (MW)",
            template="plotly_white"
        )

        st.success("✅ Prévision terminée")

        st.subheader("📊 Résultats")
        st.plotly_chart(fig_dakar, use_container_width=True)
        st.plotly_chart(fig_region, use_container_width=True)
        st.plotly_chart(fig_total, use_container_width=True)

        st.subheader("📋 Tableau des valeurs")
        st.dataframe(df_res.style.format(precision=1))

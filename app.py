import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Prédiction santé mentale étudiante", page_icon="🧠")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, encoders, feature_cols

model, encoders, feature_cols = load_artifacts()

st.title("🧠 Prédiction de risque de trouble de santé mentale")
st.write(
    "Ce modèle estime si un·e étudiant·e présente un risque de trouble de santé mentale "
    "(dépression, anxiété ou crise de panique regroupées en une seule cible), à partir "
    "de quelques informations sur son profil."
)
st.warning(
    "⚠️ Outil pédagogique — ne remplace pas un diagnostic ou un avis professionnel. "
    "Modèle entraîné sur un petit échantillon auto-déclaré."
)

gender_opts = list(encoders["gender"].classes_)
course_opts = list(encoders["course"].classes_)
year_opts = list(encoders["year_of_study"].classes_)
marital_opts = list(encoders["marital_status"].classes_)

with st.form("prediction_form"):
    gender = st.selectbox("Genre", gender_opts)
    age = st.number_input("Âge", min_value=15, max_value=40, value=20)
    course = st.selectbox("Filière", course_opts)
    year_of_study = st.selectbox("Année d'étude", year_opts)
    cgpa_numeric = st.slider("CGPA (moyenne)", 2.0, 4.0, 3.25, step=0.01)
    marital_status = st.selectbox("Statut marital", marital_opts)
    submitted = st.form_submit_button("Prédire")

if submitted:
    # Mêmes transformations que dans le notebook (strip / lower) avant encodage
    row = {
        "gender": encoders["gender"].transform([gender])[0],
        "age": age,
        "course": encoders["course"].transform([course.strip()])[0],
        "year_of_study": encoders["year_of_study"].transform(
            [year_of_study.strip().lower()]
        )[0],
        "cgpa_numeric": cgpa_numeric,
        "marital_status": encoders["marital_status"].transform([marital_status])[0],
    }
    X_new = pd.DataFrame([row])[feature_cols]

    pred = model.predict(X_new)[0]
    proba = (
        model.predict_proba(X_new)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    if pred == 1:
        st.error("⚠️ Risque détecté de trouble de santé mentale.")
    else:
        st.success("✅ Pas de risque détecté par le modèle.")

    if proba is not None:
        st.metric("Probabilité estimée", f"{proba * 100:.1f} %")

st.caption(
    "Cible prédite : mental_health_issue = dépression OU anxiété OU crise de panique."
)

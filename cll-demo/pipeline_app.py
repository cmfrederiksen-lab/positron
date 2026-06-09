import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import plotly.express as px

# --- KONFIGURATION ---
st.set_page_config(page_title="ML Pipeline Explorer", page_icon="⚙️", layout="wide")

# --- DATA INDLÆSNING & FEATURE ENGINEERING (Cached for hastighed) ---
@st.cache_data
def load_and_engineer_data():
    # 1. Indlæs
    df_car = pd.read_parquet("data/cancerregisteret.parquet")
    df_lpr = pd.read_parquet("data/landspatientregisteret.parquet")
    df_lab = pd.read_parquet("data/klinisk_biokemi.parquet")

    # 2. LABKA Features
    df_lab_merged = df_lab.merge(df_car[['patient_id', 'diagnosedato']], on='patient_id')
    df_lab_merged['dage_siden_dx'] = (df_lab_merged['proevedato'] - df_lab_merged['diagnosedato']).dt.days
    df_alc = df_lab_merged[df_lab_merged['analyse_kode'] == 'NPU02636'].copy()

    def calculate_slope(group):
        if len(group) < 2: return 0.0
        return float(np.polyfit(group['dage_siden_dx'], group['resultat'], 1)[0])

    lab_features = df_alc.groupby('patient_id').agg(
        alc_max=('resultat', 'max'),
        alc_min=('resultat', 'min'),
        alc_antal_proever=('resultat', 'count')
    ).reset_index()
    
    slopes = df_alc.groupby('patient_id').apply(calculate_slope).reset_index(name='alc_slope')
    lab_features = lab_features.merge(slopes, on='patient_id')

    # 3. LPR Features
    lpr_features = df_lpr.groupby('patient_id').size().reset_index(name='lpr_antal_besoeg')

    # 4. Saml Matrix
    df_model = df_car[['patient_id', 'koen', 'alder_ved_diagnose', 'progression_speed_LATENT']].copy()
    df_model = df_model.merge(lab_features, on='patient_id', how='left')
    df_model = df_model.merge(lpr_features, on='patient_id', how='left')
    
    df_model['alc_antal_proever'] = df_model['alc_antal_proever'].fillna(0)
    df_model['lpr_antal_besoeg'] = df_model['lpr_antal_besoeg'].fillna(0)

    return df_model

try:
    df_model = load_and_engineer_data()
except FileNotFoundError:
    st.error("Datafiler ikke fundet. Kør datagenererings-scriptet først.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("⚙️ ML Pipeline")
st.sidebar.markdown("Følg dataens rejse fra rå features til færdig model.")
trin = st.sidebar.radio(
    "Vælg procestrin:",
    ("1. Feature Matrix (Før Pre-processing)", 
     "2. Pre-processing (Samlebåndet)", 
     "3. Model Træning & Resultat",
     "4. Dokumentation (Markdown)")
)

# Klargør X og y til brug på tværs af faner
y = df_model['progression_speed_LATENT']
X = df_model.drop(columns=['patient_id', 'progression_speed_LATENT'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definer preprocessor logikken (vises og bruges i Trin 2 og 3)
numeric_features = ['alder_ved_diagnose', 'alc_max', 'alc_min', 'alc_antal_proever', 'alc_slope', 'lpr_antal_besoeg']
categorical_features = ['koen']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- TRIN 1: FEATURE MATRIX ---
if trin == "1. Feature Matrix (Før Pre-processing)":
    st.title("1. Feature Matrix (Rå Features)")
    st.markdown("Her er resultatet af vores Feature Engineering. Data er samlet, så vi har **én række pr. patient**. Men data er endnu ikke klar til maskinlæring: Forskellige kolonner har vidt forskellige skalaer (alder vs. slope), og der mangler stadig værdier (NaN) for patienter uden blodprøver.")
    
    st.markdown("### `X` (Features)")
    st.dataframe(X.head(50), use_container_width=True)
    
    st.markdown("### Opsummering af problemer før pre-processing:")
    col1, col2 = st.columns(2)
    col1.error(f"Kolonner med manglende værdier (NaN): {X.isna().sum().sum()}")
    col2.warning("Kategoriske data (Køn) står som tekst ('M'/'K') og skal kodes om til tal.")

# --- TRIN 2: PRE-PROCESSING ---
elif trin == "2. Pre-processing (Samlebåndet)":
    st.title("2. Pre-processing (Scikit-Learn ColumnTransformer)")
    st.markdown("Her sender vi vores rå `X` igennem `ColumnTransformer`. Vi imputerer manglende værdier med medianen, skalerer alle numeriske kolonner til at have en middelværdi på 0 og en standardafvigelse på 1, og One-Hot Encoder kønnet.")
    
    # Kør preprocessor for at vise resultatet visuelt
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Træk feature navne ud af preprocessor
    num_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_names, cat_names])
    
    # Lav en dataframe af den transformerede matrix for at vise i Streamlit
    df_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    
    st.markdown("### `X_train` (Efter Pre-processing)")
    st.dataframe(df_transformed.head(50), use_container_width=True)
    
    st.success("Tjek dataene nu: Alle null-værdier er fjernet. Alle tal er skaleret. 'Køn' er omdannet til en binær `koen_M` kolonne (0 = K, 1 = M). Data er nu matematisk perfekt til algoritmen.")

# --- TRIN 3: MODEL TRÆNING & RESULTAT ---
elif trin == "3. Model Træning & Resultat":
    st.title("3. Random Forest Model")
    st.markdown("Vi samler preprocessor og algoritme i én samlet scikit-learn `Pipeline` og træner på de 80% træningsdata. Herunder evaluerer vi på de resterende 20% testdata.")
    
    # Byg og træn modellen
    clf_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    with st.spinner("Træner Random Forest model..."):
        clf_rf.fit(X_train, y_train)
        y_prob = clf_rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC Score (Test Sæt)", f"{auc:.3f}")
    col2.info("En score tættere på 1.0 betyder, at modellen er fremragende til at skelne mellem hurtig og stabil progression.")
    
    st.markdown("---")
    st.markdown("### Hvad kiggede modellen efter? (Feature Importance)")
    st.markdown("Random Forest lader os kigge 'under kølerhjelmen' for at se, hvilke aggregerede features der betød mest for dens forudsigelser.")
    
    # Udtræk feature importance
    importances = clf_rf.named_steps['classifier'].feature_importances_
    num_names = clf_rf.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out(numeric_features)
    cat_names = clf_rf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_names, cat_names])
    
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_importance = df_importance.sort_values(by='Importance', ascending=True)
    
    fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# --- TRIN 4: DOKUMENTATION ---
elif trin == "4. Dokumentation (Markdown)":
    st.title("4. Teorien bag pipelinen")
    st.markdown("Herunder er gennemgangen af Python-scriptets logik, indlæst direkte fra `forklaring_train_models.md`.")
    st.markdown("---")
    
    try:
        with open("forklaring_train_models.md", "r", encoding="utf-8") as file:
            md_content = file.read()
        st.markdown(md_content)
    except FileNotFoundError:
        st.error("Kunne ikke finde filen 'forklaring_train_models.md'. Sørg for at den ligger i samme mappe som dette script.")


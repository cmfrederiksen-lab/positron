import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- KONFIGURATION ---
st.set_page_config(page_title="CLL Kohorte Explorer", page_icon="🧬", layout="wide")

# --- DATA INDLÆSNING (Med Caching for hastighed) ---
@st.cache_data
def load_data():
    car = pd.read_parquet("data/cancerregisteret.parquet")
    lpr = pd.read_parquet("data/landspatientregisteret.parquet")
    lab = pd.read_parquet("data/klinisk_biokemi.parquet")
    return car, lpr, lab

try:
    df_car, df_lpr, df_lab = load_data()
except FileNotFoundError:
    st.error("Datafiler ikke fundet. Kør datagenererings-scriptet først for at oprette Parquet-filerne i data/ mappen.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🧬 Syntetisk CLL Kohorte")
st.sidebar.markdown("Vælg datasæt til Exploratory Data Analysis (EDA):")

view_option = st.sidebar.radio(
    "Gå til:",
    ("1. Cancerregisteret (CAR)", "2. Landspatientregisteret (LPR)", "3. Klinisk Biokemi (LABKA)")
)

st.sidebar.markdown("---")
st.sidebar.info("Dette dashboard demonstrerer præ-modellering EDA af longitudinelle sundhedsdata før anvendelse af scikit-learn.")

# --- VISNING: CANCERREGISTERET ---
if view_option == "1. Cancerregisteret (CAR)":
    st.title("Cancerregisteret (Baseline Demografi)")
    
    # Nøgletal
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patienter", len(df_car))
    col2.metric("Gennemsnitsalder", f"{df_car['alder_ved_diagnose'].mean():.1f} år")
    col3.metric("Hurtig Progression (Latent)", f"{(df_car['progression_speed_LATENT'].mean() * 100):.1f} %")
    
    st.markdown("### Aldersfordeling ved diagnose")
    fig_age = px.histogram(df_car, x="alder_ved_diagnose", color="koen", barmode="overlay", 
                           title="Aldersfordeling opdelt på køn",
                           labels={"alder_ved_diagnose": "Alder", "koen": "Køn"})
    st.plotly_chart(fig_age, use_container_width=True)
    
    st.markdown("### Rå Data (Uddrag)")
    st.dataframe(df_car.head(100))

# --- VISNING: LANDSPATIENTREGISTERET ---
elif view_option == "2. Landspatientregisteret (LPR)":
    st.title("Landspatientregisteret (Ambulante Forløb)")
    
    st.markdown("Dette datasæt indeholder de løbende ambulante besøg. Patienter med hurtig progression vil typisk have en højere kontaktfrekvens.")
    
    # Beregn besøg per patient (Rettet linje)
    besoeg_per_patient = df_lpr.groupby('patient_id').size().reset_index(name='antal_besoeg')
    df_car_merged = df_car.merge(besoeg_per_patient, on='patient_id', how='left')
    
    fig_visits = px.box(df_car_merged, x="progression_speed_LATENT", y="antal_besoeg", 
                        color="progression_speed_LATENT",
                        title="Antal ambulante besøg vs. Progressionshastighed",
                        labels={"progression_speed_LATENT": "Latent Progression (0=Stabil, 1=Hurtig)", "antal_besoeg": "Antal besøg"})
    st.plotly_chart(fig_visits, use_container_width=True)
    
    st.markdown("### Rå Data (Uddrag)")
    st.dataframe(df_lpr.head(100))

# --- VISNING: KLINISK BIOKEMI ---
elif view_option == "3. Klinisk Biokemi (LABKA)":
    st.title("Klinisk Biokemi (Longitudinelle Biomarkører)")
    
    st.markdown("Her leder vi efter det kliniske signal, som vores maskinlæringsmodel senere skal fange. Vi visualiserer lymfocyttallet (ALC) over tid.")
    
    # Klargør data til scatterplot
    df_lab_alc = df_lab[df_lab['analyse_kode'] == 'NPU02636'].copy()
    df_merged_scatter = df_lab_alc.merge(df_car[['patient_id', 'progression_speed_LATENT', 'diagnosedato']], on='patient_id')
    df_merged_scatter['dage_siden_dx'] = (df_merged_scatter['proevedato'] - df_merged_scatter['diagnosedato']).dt.days
    
    # Toggle til at skjule/vise den latente variabel
    show_latent = st.checkbox("Vis farvekodning for det skjulte progressions-signal", value=True)
    
    color_col = "progression_speed_LATENT" if show_latent else None
    
    fig_lab = px.scatter(df_merged_scatter, x="dage_siden_dx", y="resultat", color=color_col,
                         trendline="ols", 
                         title="Lymfocyttaludvikling fra Diagnosetidspunkt",
                         labels={"dage_siden_dx": "Dage siden diagnose", "resultat": "Lymfocyttal (10^9/L)", "progression_speed_LATENT": "Progression"},
                         opacity=0.3)
    st.plotly_chart(fig_lab, use_container_width=True)
    
    # --- FORDELING AF LABORATORIESVAR PR KØN OG PROGNOSE ---
    st.markdown("---")
    st.markdown("### Fordeling af laboratoriesvar pr. køn og prognose")
    st.markdown("Her undersøger vi, om der er en systematisk forskel på biomarkør-niveauerne mellem stabil og hurtig progression, opdelt på køn.")
    
    # Merge data for at koble køn og prognose på lab-svarene
    df_lab_demografi = df_lab.merge(
        df_car[['patient_id', 'koen', 'progression_speed_LATENT']], 
        on='patient_id'
    )
    
    # Gør prognose-variablen mere læsbar til vores plot
    df_lab_demografi['Prognose'] = df_lab_demografi['progression_speed_LATENT'].map(
        {0: 'Stabil (0)', 1: 'Hurtig (1)'}
    )
    
    # Dropdown til at vælge analyse
    unike_analyser = df_lab_demografi['analyse_navn'].unique()
    valgt_analyse = st.selectbox("Vælg biokemisk analyse for at se fordeling:", unike_analyser)
    
    # Filtrer data til den valgte analyse
    df_plot = df_lab_demografi[df_lab_demografi['analyse_navn'] == valgt_analyse]
    
    # Byg Boxplot med Plotly
    fig_dist = px.box(
        df_plot, 
        x="koen", 
        y="resultat", 
        color="Prognose",
        title=f"Fordeling af {valgt_analyse}",
        labels={
            "koen": "Køn", 
            "resultat": f"Målt værdi", 
            "Prognose": "Latent Progression"
        },
        color_discrete_sequence=["#1f77b4", "#d62728"] # Sikrer tydelig kontrast
    )
    
    # Optimer layout
    fig_dist.update_layout(boxmode='group') 
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- RÅ DATA ---
    st.markdown("---")
    st.markdown("### Rå Data (Uddrag)")
    st.dataframe(df_lab.head(100))


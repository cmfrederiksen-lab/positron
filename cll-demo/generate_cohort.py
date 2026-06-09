import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

# Sæt et seed for reproducerbarhed
np.random.seed(42)

# --- 1. PYDANTIC SKEMAER (Datakvalitet & Validering) ---

class CancerRegistryRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    
    patient_id: str
    koen: Literal['M', 'K']
    alder_ved_diagnose: int = Field(ge=18, le=120)
    diagnosedato: datetime
    icd10_primær: str
    progression_speed_LATENT: Literal[0, 1]

class LPRContact(BaseModel):
    dw_ek_kontakt: str
    patient_id: str
    dato_ind: datetime
    kontakttype: Literal['Ambulant', 'Indlagt']
    p_diagnose: str

class LabResult(BaseModel):
    patient_id: str
    proevedato: datetime
    analyse_kode: str
    analyse_navn: str
    resultat: float = Field(ge=0.0) # Sikrer at vi aldrig får negative blodprøvesvar
    enhed: str

# --- 2. GENERERINGSFUNKTIONER ---

start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 12, 31)

def generate_cancer_registry(n: int) -> pd.DataFrame:
    records = []
    date_range = (end_date - start_date).days
    
    for i in range(1, n + 1):
        # Generer rå data
        diagnosedato = start_date + timedelta(days=np.random.randint(0, date_range))
        prog_speed = int(np.random.choice([0, 1], p=[0.75, 0.25]))
        alder = int(np.clip(np.random.normal(loc=70, scale=9), 40, 95))
        koen = str(np.random.choice(['M', 'K'], p=[0.6, 0.4]))
        
        # Pydantic validerer dataene ved instansiering
        record = CancerRegistryRecord(
            patient_id=f"SYNT-{str(i).zfill(5)}",
            koen=koen,
            alder_ved_diagnose=alder,
            diagnosedato=diagnosedato,
            icd10_primær='DC911',
            progression_speed_LATENT=prog_speed
        )
        records.append(record.model_dump())
        
    return pd.DataFrame(records)

def generate_lpr_contacts(df_car: pd.DataFrame) -> pd.DataFrame:
    records = []
    
    for _, row in df_car.iterrows():
        n_visits = np.random.randint(2, 15) if row['progression_speed_LATENT'] == 0 else np.random.randint(10, 30)
        
        for _ in range(n_visits):
            days_after_dx = np.random.randint(0, 1800)
            visit_date = row['diagnosedato'] + timedelta(days=days_after_dx)
            
            record = LPRContact(
                dw_ek_kontakt=f"KONT-{np.random.randint(100000, 999999)}",
                patient_id=row['patient_id'],
                dato_ind=visit_date,
                kontakttype='Ambulant',
                p_diagnose='DC911'
            )
            records.append(record.model_dump())
            
    df_lpr = pd.DataFrame(records)
    return df_lpr.sort_values(by=['patient_id', 'dato_ind']).reset_index(drop=True)

def generate_lab_data(df_car: pd.DataFrame, df_lpr: pd.DataFrame) -> pd.DataFrame:
    records = []
    
    for _, row in df_lpr.iterrows():
        patient = df_car[df_car['patient_id'] == row['patient_id']].iloc[0]
        days_since_dx = (row['dato_ind'] - patient['diagnosedato']).days
        
        if days_since_dx < 0: continue
        
        # Simuler Lymfocyttal
        base_alc = np.random.normal(15, 5)
        stigning = np.random.uniform(0.01, 0.05) if patient['progression_speed_LATENT'] == 1 else np.random.uniform(0.00, 0.005)
        alc_val = max(1.0, base_alc + (days_since_dx * stigning) + np.random.normal(0, 2))
        
        # Simuler Beta-2-mikroglobulin
        b2m_val = max(0.5, np.random.normal(2.5, 0.5) + (alc_val * 0.05))
        
        records.append(LabResult(
            patient_id=row['patient_id'],
            proevedato=row['dato_ind'],
            analyse_kode='NPU02636',
            analyse_navn='Lymfocytter [ALC]',
            resultat=round(alc_val, 2),
            enhed='10^9/L'
        ).model_dump())
        
        records.append(LabResult(
            patient_id=row['patient_id'],
            proevedato=row['dato_ind'],
            analyse_kode='NPU01358',
            analyse_navn='Beta-2-mikroglobulin',
            resultat=round(b2m_val, 2),
            enhed='mg/L'
        ).model_dump())
        
    return pd.DataFrame(records)

# --- 3. EKSEKVERING ---

import os

if __name__ == "__main__":
    n_patients = 1500
    print(f"Genererer syntetisk kohorte for {n_patients} patienter...")
    
    df_car = generate_cancer_registry(n_patients)
    print("CAR-data genereret og valideret via Pydantic.")
    
    df_lpr = generate_lpr_contacts(df_car)
    print("LPR-data genereret og valideret via Pydantic.")
    
    df_lab = generate_lab_data(df_car, df_lpr)
    print("LABKA-data genereret og valideret via Pydantic.")
    
    # Validering af outputstruktur
    print("\nEksempel på Lab Resultater:")
    print(df_lab.head())


# --- 3. EKSEKVERING OG GEM TIL DISK ---

    n_patients = 10000
    print(f"Genererer syntetisk kohorte for {n_patients} patienter...")
    
    df_car = generate_cancer_registry(n_patients)
    print("CAR-data genereret og valideret.")
    
    df_lpr = generate_lpr_contacts(df_car)
    print("LPR-data genereret og valideret.")
    
    df_lab = generate_lab_data(df_car, df_lpr)
    print("LABKA-data genereret og valideret.")
    
    # Opret data-mappe hvis den ikke eksisterer
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGemmer tabeller til disk (Parquet-format)...")
    
    # Gemmer DataFrames som Parquet-filer
    df_car.to_parquet(os.path.join(output_dir, "cancerregisteret.parquet"), index=False)
    df_lpr.to_parquet(os.path.join(output_dir, "landspatientregisteret.parquet"), index=False)
    df_lab.to_parquet(os.path.join(output_dir, "klinisk_biokemi.parquet"), index=False)
    
    print(f"Succes! Alle tabeller er gemt i mappen: ./{output_dir}/")


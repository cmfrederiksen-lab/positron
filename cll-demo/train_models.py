import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("1. Indlæser data fra Parquet-filer...")
df_car = pd.read_parquet("data/cancerregisteret.parquet")
df_lpr = pd.read_parquet("data/landspatientregisteret.parquet")
df_lab = pd.read_parquet("data/klinisk_biokemi.parquet")

# --- FEATURE ENGINEERING ---
print("2. Bygger feature matrix...")

# A. Håndtering af Klinisk Biokemi (LABKA) - Lymfocytter
# Find dage siden diagnose for hver prøve
df_lab_merged = df_lab.merge(df_car[['patient_id', 'diagnosedato']], on='patient_id')
df_lab_merged['dage_siden_dx'] = (df_lab_merged['proevedato'] - df_lab_merged['diagnosedato']).dt.days

# Filtrer kun lymfocytter (ALC)
df_alc = df_lab_merged[df_lab_merged['analyse_kode'] == 'NPU02636'].copy()

# Funktion til at udregne hældningskoefficient for en enkelt patient
def calculate_slope(group):
    # Hvis patienten har mindre end 2 prøver, kan vi ikke beregne en hældning
    if len(group) < 2:
        return 0.0
    x = group['dage_siden_dx']
    y = group['resultat']
    # np.polyfit af grad 1 returnerer [hældning, skæring]
    return float(np.polyfit(x, y, 1)[0])

# Aggreger max, min og tæl antal prøver
lab_features = df_alc.groupby('patient_id').agg(
    alc_max=('resultat', 'max'),
    alc_min=('resultat', 'min'),
    alc_antal_proever=('resultat', 'count')
).reset_index()

# Udregn og tilføj hældningskoefficienten
slopes = df_alc.groupby('patient_id').apply(calculate_slope).reset_index(name='alc_slope')
lab_features = lab_features.merge(slopes, on='patient_id')

# B. Håndtering af Landspatientregisteret (LPR)
# Tæl antal ambulante besøg pr. patient
lpr_features = df_lpr.groupby('patient_id').size().reset_index(name='lpr_antal_besoeg')

# C. Saml den endelige Matrix (Baseret på CAR demografi)
df_model = df_car[['patient_id', 'koen', 'alder_ved_diagnose', 'progression_speed_LATENT']].copy()

# Join de nye features på (Left join, da nogle patienter evt. mangler lab-svar)
df_model = df_model.merge(lab_features, on='patient_id', how='left')
df_model = df_model.merge(lpr_features, on='patient_id', how='left')

# Udfyld manglende værdier i antal besøg/prøver med 0
df_model['alc_antal_proever'] = df_model['alc_antal_proever'].fillna(0)
df_model['lpr_antal_besoeg'] = df_model['lpr_antal_besoeg'].fillna(0)

# --- DEFINER X OG Y ---
print("3. Klargør til Maskinlæring...")
# Y er vores target. X er alle features (undtagen target og patient_id)
y = df_model['progression_speed_LATENT']
X = df_model.drop(columns=['patient_id', 'progression_speed_LATENT'])

# Split data i træning (80%) og test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- SCIKIT-LEARN PIPELINES ---
# Vi opdeler vores pre-processing så numeriske og kategoriske variabler behandles forskelligt
numeric_features = ['alder_ved_diagnose', 'alc_max', 'alc_min', 'alc_antal_proever', 'alc_slope', 'lpr_antal_besoeg']
categorical_features = ['koen']

# Numerisk transformer: Imputerer manglende værdier (med median) og skalerer (StandardScaler)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Kategorisk transformer: One-hot encoder kønnet ('M', 'K' -> 0 og 1)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first')) # drop='first' undgår multikollinearitet
])

# Saml pre-processing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model 1: Logistic Regression
clf_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Model 2: Random Forest
clf_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# --- TRÆNING OG EVALUERING ---
print("\n--- Resultater: Logistic Regression ---")
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
y_prob_lr = clf_lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_lr):.3f}")

print("\n--- Resultater: Random Forest ---")
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.3f}")



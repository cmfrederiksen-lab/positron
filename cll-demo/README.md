# 🧬 Syntetisk Prædiktion af Sygdomsprogression i Lymfoide Cancere (CLL)

Dette projekt er en komplet, *end-to-end* maskinlæringspipeline bygget i Python. Projektet demonstrerer evnen til at omsætte kompleks, longitudinel sundhedsdata (inspireret af danske nationale registre) til prædiktive modeller ved hjælp af moderne data science værktøjer.

Projektet bygger bro mellem klassisk klinisk datamanagement, probabilistisk modellering og moderne `scikit-learn` AI-workflows.

## 🎯 Formål og Domæneviden
Når man arbejder med registerdata, er den største udfordring sjældent selve algoritmen, men derimod *feature engineering* – at udtrække det kliniske signal fra støjende, longitudinelle observationer. 

Dette repository simulerer en fuld arbejdsgang for en kohorte af CLL-patienter og afspejler de rigtige strukturer fra:
* **Cancerregisteret (CAR):** Baseline demografi og diagnosetidspunkt.
* **Landspatientregisteret (LPR):** Ambulante kontaktmønstre.
* **Klinisk Biokemi (LABKA/BDS):** Longitudinelle biomarkører (inkl. ægte NPU-koder som `NPU02636` for lymfocytter), der indeholder et "latent signal" for sygdomsprogression.

## 🛠️ Tech Stack
Projektet udnytter en moderne, hurtig og typesikker Python-stack, optimeret til lokal afvikling (f.eks. på Apple Silicon):
* **Package Management:** `uv` for lynhurtig virtuel miljøstyring.
* **Datakvalitet:** `Pydantic` til skemavalidering under datagenerering.
* **Data Manipulation:** `pandas` og `numpy` med lagring i `.parquet` format for at bevare datatyper.
* **Maskinlæring:** `scikit-learn` (Pipelines, ColumnTransformers, Random Forest, Logistic Regression).
* **Visualisering & UI:** `streamlit` og `plotly` til interaktive, webbaserede dashboards.

---

## 📂 Projektstruktur

| Fil | Beskrivelse |
| :--- | :--- |
| `generate_cohort.py` | Bygger den syntetiske kohorte fra bunden. Validerer data strict via Pydantic og gemmer i Parquet-format. |
| `app.py` | Streamlit Dashboard til **Exploratory Data Analysis (EDA)**. Visualiserer fordelinger og tidsmæssige trends før ML. |
| `train_models.py` | Core ML-script. Udfører domænespecifik feature engineering (bl.a. udregning af hældningskoefficienter via `polyfit`) og træner to prædiktive modeller. |
| `pipeline_app.py` | Pædagogisk Streamlit Dashboard, der trin-for-trin forklarer og visualiserer transformationsprocessen fra rå data til færdig scikit-learn model. |
| `forklaring_train_models.md` | Dokumentation af teorien og de matematiske valg bag maskinlæringspipelinen. |

---

## 🚀 Kom Godt I Gang (Lokal Eksekvering)

### 1. Opsætning af miljø
Projektet bruger `uv` til at håndtere pakker og virtuelle miljøer lynhurtigt.

```bash
# Klon repository og gå til mappen
cd cll-demo

# Opret et virtuelt miljø
uv venv

# Aktiver miljøet (Mac/Linux)
source .venv/bin/activate

# Installer alle nødvendige afhængigheder
uv add pandas numpy pydantic scikit-learn pyarrow streamlit plotly statsmodels jupyter




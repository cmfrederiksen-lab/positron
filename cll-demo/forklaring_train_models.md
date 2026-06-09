# Gennemgang af train_models.py (Scikit-Learn for Begyndere)

Dette dokument forklarer trin for trin, hvordan maskinlærings-script transformerer rå registerdata til to prædiktive modeller (Logistic Regression og Random Forest). 

Scriptet er bygget op i logiske faser: Import, Dataindlæsning, Feature Engineering, Klargøring, Pre-processing og til sidst Træning/Evaluering.

## 1. Import af værktøjer (Biblioteker)
I Python bruger vi eksterne biblioteker til at gøre det tunge arbejde. 
* **`pandas` og `numpy`:** Det absolutte fundament for datahåndtering og matematik i Python.
* **`sklearn` (Scikit-Learn):** Pythons standardbibliotek til maskinlæring. Vi importerer specifikke byggeklodser herfra: funktioner til at dele data op (`train_test_split`), bygge samlebånd (`Pipeline`), transformere data (`StandardScaler`, `OneHotEncoder`) og selve algoritmerne (`RandomForestClassifier` osv.).

## 2. Indlæsning af Data
Vi indlæser vores tre tabeller fra de Parquet-filer, vi genererede tidligere. Parquet er fantastisk, fordi det husker alle datatyper (som datoer og tal) helt præcist i modsætning til CSV-filer.

## 3. Feature Engineering (At bygge "features")
Dette er hjertet i enhver god maskinlæringsmodel. Algoritmer forstår ikke longitudinelle data (flere prøver over tid). De har brug for én række pr. patient med opsummerede tal (features).

* **Tidsberegning:** Først kobler vi diagnosedatoen fra Cancerregisteret sammen med blodprøverne for at regne ud, hvor mange dage efter diagnosen prøven er taget (`dage_siden_dx`).
* **Filtrering:** Vi isolerer den specifikke blodprøve (Lymfocytter), vi vil analysere.
* **Hældningskoefficient (Slope):** Funktionen `calculate_slope` tager alle en patients blodprøver og trækker en matematisk linje igennem dem ved hjælp af `np.polyfit`. Den udregner linjens hældning $a$ i den klassiske ligning $y = ax + b$. En høj hældning betyder, at tallet stiger hurtigt.
* **Aggregering:** Vi bruger `groupby` til at regne max-værdi, min-værdi og antallet af prøver ud for hver patient. Dette er præcis samme logik som `GROUP BY` i SQL.
* **Samling:** Til sidst klistrer vi alle vores nye features sammen i én stor hovedtabel (`df_model`). Hvis en patient mangler laboratoriesvar eller ambulante besøg, udfylder vi de tomme felter med 0.

## 4. Klargøring til Maskinlæring (X og y)
Maskinlæring handler om at finde sammenhængen mellem input og output.
* **`y` (Target):** Dette er facitlisten. Det er den kolonne (`progression_speed_LATENT`), vi gerne vil have modellen til at forudsige.
* **`X` (Features):** Dette er alle input-variablerne (alder, køn, blodprøve-hældning). Vi fjerner `patient_id` (da et ID ikke har nogen prædiktiv værdi) og selve facitlisten fra denne gruppe.
* **Train/Test Split:** Vi deler vores data i to bunker. 80% bruges til at træne modellen, og 20% gemmes væk som en "eksamen", modellen ikke har set før. `stratify=y` sikrer, at fordelingen af hurtig/stabil progression er ens i begge bunker.

## 5. Pre-processing (Samlebåndet)
Før algoritmerne kan tygge på dataene, skal de formateres helt perfekt. Vi bygger en `Pipeline` (et samlebånd), der gør dette automatisk:

* **Numeriske variabler (Tal):** * `SimpleImputer` udfylder eventuelle manglende værdier (hvis en patient f.eks. kun havde én blodprøve og derved mangler en hældning) med medianen af alle patienter.
  * `StandardScaler` skalerer tallene. En hældning på 0.05 og en alder på 75 er meget forskellige størrelser. Skaleringen sætter alle tal på samme målestok, så en algoritme som Logistic Regression ikke bliver forvirret.
* **Kategoriske variabler (Tekst):** * Computere forstår ikke bogstaver som 'M' og 'K'. `OneHotEncoder` laver kønnet om til et 0 eller 1-tal.
* **ColumnTransformer:** Denne funktion samler de to samlebånd og anvender den numeriske behandling på tallene og den kategoriske behandling på teksten.

## 6. Træning og Evaluering
Nu bliver modellerne testet på den syntetiske virkelighed.

* **Logistic Regression:** En klassisk statistisk metode. Den er simpel, hurtig og meget nem at fortolke (man kan trække "Odds Ratios" ud af den).
* **Random Forest:** En "skov" af mange forskellige beslutningstræer. Den er fremragende til at finde komplekse, ikke-lineære mønstre i data, som Logistic Regression måske overser.
* **Træning (`fit`):** Vi beder modellen kigge på `X_train` og `y_train` for at lære mønstrene.
* **Forudsigelse (`predict`):** Vi beder modellen gætte resultatet for vores eksamenssæt (`X_test`).
* **Evaluering:** * `classification_report` viser os, hvor præcis modellen er til at ramme de to klasser.
  * `ROC-AUC Score` er et enkelt tal mellem 0 og 1, der beskriver modellens overordnede evne til at kende forskel på stabil og hurtig progression. En score på 0.5 er et rent gæt, mens 1.0 er perfektion.



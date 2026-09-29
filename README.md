# LibrettiEx OCR — pipeline OCR e analisi strutturata dei campi

LibrettiEx OCR è una pipeline Python per trascrivere immagini di documenti storici tramite modelli Vision/OCR eseguiti localmente con Ollama, normalizzare le trascrizioni, valutarle rispetto a una ground truth manuale e analizzare separatamente i campi strutturali dei libretti.

Il repository contiene soltanto il codice, i template strutturali e il dizionario esterno necessari alla pipeline. Il dataset, le trascrizioni manuali, gli output dei modelli e i risultati sperimentali non sono pubblicati e devono essere creati localmente.

## Struttura del repository

```text
project_librettiEx_ocr/
├── run_ocr_models.py
├── normalize_texts.py
├── evaluate_normalized.py
├── field_analysis/
│   ├── parse_fields.py
│   ├── evaluate_fields.py
│   └── templates/
│       ├── template1.yaml
│       ├── template2.yaml
│       ├── template3.yaml
│       └── dictionaries.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

### Script principali

- `run_ocr_models.py`: esegue i modelli Vision/OCR tramite le API locali di Ollama e salva gli output grezzi.
- `normalize_texts.py`: normalizza ground truth e trascrizioni OCR.
- `evaluate_normalized.py`: calcola le metriche globali di trascrizione, tra cui CER e WER.
- `field_analysis/parse_fields.py`: ricostruisce la struttura key-value dei libretti usando template noti, senza consultare la ground truth.
- `field_analysis/evaluate_fields.py`: confronta OCR e ground truth campo per campo e produce metriche PRE/POST.
- `field_analysis/templates/`: contiene i template strutturali e il dizionario esterno usato per i campi correggibili.

## Requisiti

- Python 3.12 o versione compatibile;
- Ollama installato e in esecuzione;
- almeno un modello Vision/OCR disponibile in Ollama;
- `rapidfuzz`;
- `PyYAML`.

Installazione delle dipendenze:

```bash
python -m pip install -r requirements.txt
```

## Installazione

Clonare il repository:

```bash
git clone https://github.com/alessiopgg/project_librettiEx_ocr.git
cd project_librettiEx_ocr
```

Creare un ambiente virtuale:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Su Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cartelle da creare localmente

Il dataset non è incluso nel repository.

Creare:

```text
data/
├── screenshot/
└── text_visionato/
```

Su Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data\screenshot, data\text_visionato
```

La struttura locale completa si sviluppa quindi come:

```text
project_librettiEx_ocr/
├── data/
│   ├── screenshot/          # immagini dei libretti
│   └── text_visionato/      # ground truth manuali
├── output/                  # creato dalle esecuzioni OCR
├── evaluation/              # creato dalla valutazione globale
├── field_analysis/
│   ├── parsed/              # output del parser strutturale
│   └── results/             # risultati della field analysis
└── ...
```

Le cartelle contenenti dati e risultati sono escluse dalla pubblicazione Git.

## Preparazione dei dati

Inserire le immagini in:

```text
data/screenshot/
```

Sono supportati i formati PNG, JPG, JPEG, WEBP, BMP, GIF, TIFF e TIF.

Per la valutazione, ogni immagine deve avere una trascrizione manuale con lo stesso nome di base:

```text
data/screenshot/libretto_001.png
data/text_visionato/libretto_001.txt
```

La ground truth deve riprodurre il testo del documento senza correggere o completare il contenuto originale.

---

# 1. Esecuzione OCR

## Configurazione di Ollama

Verificare l'installazione:

```bash
ollama --version
```

Esempio di modello:

```bash
ollama pull qwen3-vl:32b-instruct
```

Altri modelli previsti dallo script:

```bash
ollama pull gemma4:31b
ollama pull nemotron3:33b
ollama pull qwen3.5:27b
ollama pull deepseek-ocr:latest
ollama pull glm-ocr:latest
```

Non è necessario installarli tutti.

Avviare Ollama, per esempio con una finestra di contesto da 8192 token:

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

Su Windows PowerShell:

```powershell
$env:OLLAMA_CONTEXT_LENGTH = "8192"
ollama serve
```

## Eseguire un modello

```bash
python run_ocr_models.py \
  --models qwen3-vl:32b-instruct \
  --num-ctx 8192 \
  --num-predict 4096 \
  --description "Trascrizione con Qwen3-VL"
```

Per confrontare più modelli:

```bash
python run_ocr_models.py \
  --models qwen3-vl:32b-instruct gemma4:31b nemotron3:33b qwen3.5:27b \
  --num-ctx 8192 \
  --num-predict 4096 \
  --description "Confronto modelli Vision"
```

Per una prova rapida:

```bash
python run_ocr_models.py \
  --models qwen3-vl:32b-instruct \
  --limit 1 \
  --description "Test rapido"
```

Ogni nuova esecuzione crea una cartella numerata:

```text
output/
└── experiment_001/
    ├── experiment.json
    └── qwen3-vl_32b-instruct/
        ├── prompt.txt
        ├── metadata.jsonl
        ├── libretto_001.txt
        └── ...
```

Le trascrizioni vengono conservate nella forma restituita dal modello, senza normalizzazione o correzione.

Per riprendere un esperimento:

```bash
python run_ocr_models.py \
  --models qwen3-vl:32b-instruct \
  --experiment-id 1
```

I file già presenti vengono saltati; `--overwrite` permette di rigenerarli.

---

# 2. Normalizzazione

Dopo avere prodotto gli output OCR e inserito le ground truth:

```bash
python normalize_texts.py --clean
```

Lo script legge:

```text
data/text_visionato/libretto_*.txt
output/experiment_*/<modello>/libretto_*.txt
```

e genera:

```text
evaluation/normalized/
├── references/
├── hypotheses/
└── normalization_report.json
```

La normalizzazione applica:

- rimozione dell'eventuale BOM;
- Unicode NFC;
- `casefold()` per ignorare differenze tra maiuscole e minuscole;
- sostituzione di punteggiatura e simboli con spazi;
- collasso degli spazi multipli;
- conservazione di lettere accentate e numeri;
- rimozione del markup `<unclear>` mantenendone il contenuto;
- rimozione di `<gap/>` e `<illegible/>`;
- conservazione dell'ordine delle parole.

I file originali non vengono modificati.

---

# 3. Valutazione globale

Dopo la normalizzazione:

```bash
python evaluate_normalized.py --clean
```

Lo script produce:

```text
evaluation/metrics/normalized/
├── per_file_metrics.csv
├── model_summary.csv
└── metrics_report.json
```

La valutazione globale utilizza, tra le altre, metriche basate sulla distanza di Levenshtein:

- **CER (Character Error Rate)**;
- **WER (Word Error Rate)**.

CER e WER considerano sostituzioni, cancellazioni e inserzioni rispettivamente a livello di carattere e di parola.

---

# 4. Analisi strutturata dei campi

La field analysis estende la valutazione globale separando il documento in coppie **chiave-valore**.

Il parser è **template-aware**: conosce il layout logico atteso e l'ordine delle chiavi prestampate, ma non utilizza la ground truth per individuare i campi dell'OCR.

## Template

I template si trovano in:

```text
field_analysis/templates/
```

Ogni template dichiara:

1. un identificativo;
2. i documenti che utilizzano quel layout;
3. i campi nell'ordine strutturale atteso;
4. per ogni campo, la chiave da cercare e la modalità di detection;
5. l'identificativo e la categoria del valore.

Esempio:

```yaml
template:
  id: template1
  documents:
    - libretto_001

  fields:
    - order: 1
      key:
        id: k01_cognome_nome
        text: "Cognome e nome"
        detection: {scope: global, match: fuzzy}
      value:
        id: cognome_nome
        category: string
```

Le categorie operative dei valori sono:

```text
string
numeric
dictionary
```

Per i campi `dictionary` viene inoltre indicato il dizionario:

```yaml
value:
  id: facolta_left
  category: dictionary
  dictionary: faculties
```

Le chiavi sufficientemente informative vengono normalmente cercate globalmente con fuzzy matching:

```yaml
detection: {scope: global, match: fuzzy}
```

Le chiavi brevi e ambigue, come `a`, `in`, `di` o `per`, vengono invece cercate solo localmente tra chiavi strutturali già individuate:

```yaml
detection: {scope: local, match: exact}
```

## Parsing strutturale

Il flusso principale di `parse_fields.py` è:

```text
testo OCR
  ↓
normalizzazione e tokenizzazione
  ↓
ricerca delle global key
  ↓
selezione della migliore sequenza coerente con l'ordine del template
  ↓
recovery delle global key mancanti
  ↓
ricerca delle local key
  ↓
estrazione dei valori
```

Soglie predefinite del parser:

```text
global_threshold       = 88
recovery_threshold     = 72
local_fuzzy_threshold  = 82
```

Esempio:

```bash
python field_analysis/parse_fields.py \
  output/experiment_005/gemma4_31b \
  --templates field_analysis/templates \
  --output field_analysis/parsed/gemma4_complete
```

Per ogni `libretto_*.txt` viene prodotto un JSON diagnostico contenente key trovate, stato del matching, valori estratti e boundary.

## Valutazione per campo

`evaluate_fields.py` esegue il parsing **indipendentemente** sulla trascrizione OCR e sulla ground truth usando gli stessi template.

La ground truth viene utilizzata soltanto nella fase di confronto, non per delimitare i campi OCR.

La valutazione distingue:

- key rilevata;
- key esattamente corretta;
- value valutabile o non valutabile;
- exact match del valore;
- CER/WER del valore;
- risultato **PRE**, prima del post-processing;
- risultato **POST**, dopo il post-processing.

Per i campi `dictionary` è possibile correggere un valore OCR non vuoto usando un dizionario esterno.

La configurazione utilizzata nella campagna finale di field analysis usa esplicitamente:

```text
dictionary_threshold = 70
dictionary_margin    = 15
```

Esempio:

```bash
python field_analysis/evaluate_fields.py \
  output/experiment_005/gemma4_31b \
  --gt-dir data/text_visionato \
  --templates field_analysis/templates \
  --dictionaries field_analysis/templates/dictionaries.yaml \
  --dictionary-threshold 70 \
  --dictionary-margin 15 \
  --output field_analysis/results/gemma4_complete
```

Nota: i default CLI del file sono `80` per la soglia dictionary e `0` per il margine; per riprodurre la campagna finale vanno quindi passati esplicitamente `70` e `15`.

Se una key necessaria alla segmentazione manca, il parser non forza artificialmente il boundary del valore. Il campo viene marcato come non valutabile. Per i campi `dictionary`, la fase POST può inoltre tentare un recupero locale controllato nella regione strutturalmente disponibile.

## Output della field analysis

Per ogni configurazione vengono prodotti:

```text
field_analysis/results/<configurazione>/
├── documents/
│   ├── libretto_001.csv
│   └── ...
├── all_fields.csv
├── field_summary.csv
├── document_summary.csv
├── category_summary.csv
├── parsed_ocr.json
├── parsed_gt.json
└── report.json
```

### File principali

- `all_fields.csv`: una riga per ogni campo strutturale di ogni documento;
- `field_summary.csv`: aggregazione per campo semantico;
- `document_summary.csv`: aggregazione per documento;
- `category_summary.csv`: metriche aggregate per `key`, `dictionary`, `numeric` e `string`, separate in PRE/POST;
- `parsed_ocr.json`: parsing completo degli output OCR;
- `parsed_gt.json`: parsing completo della ground truth;
- `report.json`: riepilogo della run e dei parametri utilizzati.

## Dizionario esterno

`field_analysis/templates/dictionaries.yaml` contiene le voci ammesse per i campi correggibili.

Nel progetto corrente viene utilizzato un dizionario di denominazioni accademiche costruito da fonti storiche e istituzionali esterne alla ground truth.

Il dizionario:

- non viene costruito dai valori dei documenti di test;
- non completa automaticamente valori OCR vuoti;
- corregge soltanto letture OCR sufficientemente vicine a una voce nota;
- utilizza una soglia di similarità e un margine rispetto al secondo candidato.

---

# Riproduzione della pipeline

Con immagini e ground truth disponibili localmente, il workflow generale è:

```text
immagini
   ↓
run_ocr_models.py
   ↓
output OCR grezzo
   ├───────────────┐
   ↓               ↓
normalize_texts.py parse_fields.py
   ↓               ↓
evaluate_normalized.py
                   ↓
            evaluate_fields.py
```

Per visualizzare tutti i parametri disponibili:

```bash
python run_ocr_models.py --help
python normalize_texts.py --help
python evaluate_normalized.py --help
python field_analysis/parse_fields.py --help
python field_analysis/evaluate_fields.py --help
```

---

# Privacy e pubblicazione

Il repository pubblico contiene soltanto codice e configurazioni necessarie alla pipeline.

Non sono pubblicati:

- immagini dei libretti;
- ground truth manuali;
- output OCR;
- file normalizzati;
- risultati delle metriche;
- JSON e CSV prodotti dalla field analysis.

Il `.gitignore` utilizza una whitelist: tutto viene ignorato di default e vengono inclusi esplicitamente soltanto gli script, il README, il file delle dipendenze e i template pubblicabili.

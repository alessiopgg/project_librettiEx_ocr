# LibrettiEx OCR

**Valutazione comparativa di modelli OCR e Vision-Language per la trascrizione di libretti universitari storici.**  
*Project Work in Data Mining — Università degli Studi di Firenze.*

Il progetto studia la trascrizione automatica di documenti storici che combinano **testo prestampato, scrittura manoscritta e layout a più colonne**. Su un dataset di **18 libretti**, corredati da trascrizioni manuali di riferimento (*ground truth*), sono stati confrontati quattro modelli Vision-Language generalisti (**Gemma 4, Qwen3-VL, Qwen3.5 e Nemotron 3**) e, separatamente, due modelli OCR specializzati (**GLM-OCR e DeepSeek-OCR**). I modelli sono stati eseguiti localmente tramite **Ollama**, confrontando differenti prompt e dimensioni della finestra di contesto.

La valutazione si articola in due livelli:

* **Trascrizione globale:** confronto dell'intero testo tramite Character Error Rate (**CER**), Word Error Rate (**WER**) e **Word F1** indipendente dall'ordine delle parole.
* **Analisi strutturata per campo:** ricostruzione delle coppie chiave–valore mediante tre template noti, valutando riconoscimento delle chiavi, copertura e accuratezza dei valori `string`, `numeric` e `dictionary`, prima (**PRE**) e dopo (**POST**) un post-processing controllato.

La pipeline separa l'esecuzione OCR, la normalizzazione, il parsing strutturale e la valutazione. **Il parser non consulta la ground truth per individuare i campi OCR**: il riferimento manuale viene utilizzato soltanto nel confronto finale.

## Risultati principali

### Qualità della trascrizione

Migliori configurazioni per ciascun modello generalista, con **contesto di 8.192 token**. Le metriche sono calcolate sulle trascrizioni normalizzate, **prima del post-processing**.

|Modello|Prompt|CER ↓|WER ↓|Word F1 ↑|
|-|-|-:|-:|-:|
|**Gemma 4 (31B)**|Completo|**1,57%**|**4,03%**|**96,46%**|
|Qwen3-VL (32B)|Essenziale|2,79%|5,73%|95,43%|
|Qwen3.5 (27B)|Intermedio|4,66%|10,22%|91,26%|
|Nemotron 3 (33B)|Completo|18,63%|25,18%|84,26%|

**Gemma 4** ottiene i risultati migliori nel benchmark principale, seguita da **Qwen3-VL**, che raggiunge prestazioni vicine con il prompt essenziale. I modelli OCR specializzati sono stati analizzati separatamente e non sono inclusi in questa graduatoria.

### Analisi strutturata dei campi

I tre template descrivono complessivamente **316 coppie chiave–valore**: 155 `string`, 125 `numeric` e 36 `dictionary`.

|Modello|Chiavi individuate|Valori valutabili (PRE)|
|-|-:|-:|
|**Gemma 4**|**315/316 (99,68%)**|**314/316 (99,37%)**|
|Qwen3-VL|315/316 (99,68%)|314/316 (99,37%)|
|Qwen3.5|313/316 (99,05%)|310/316 (98,10%)|
|Nemotron 3|273/316 (86,39%)|244/316 (77,22%)|

Per **Gemma 4**, le corrispondenze esatte dei valori valutabili sono **115/124 (92,74%)** sui campi `numeric` e **130/154 (84,42%)** sui campi `string`. Sui campi `dictionary`, la correzione tramite dizionario esterno porta i valori esatti da **32/36 (88,89%)** a **34/36 (94,44%)**. Queste percentuali sono **condizionate ai campi valutabili**, non a tutti i campi attesi.

La canonicalizzazione delle chiavi riconosciute e le correzioni dei valori `dictionary` migliorano anche le metriche globali: per Gemma 4 il **WER scende dal 4,03% al 3,76%** e per Qwen3-VL **dal 5,73% al 5,29%**. Il post-processing non ricostruisce le chiavi mancanti e non corregge automaticamente i valori `numeric` o `string`.

## Riproducibilità

Il repository contiene **codice Python, template YAML e dizionario esterno**, ma **non** pubblica le immagini dei libretti, le ground truth manuali, gli output OCR o i CSV/JSON dei risultati. Per eseguire la pipeline servono quindi documenti propri e le corrispondenti trascrizioni di riferimento.

**Requisiti:** Python 3.10+ (consigliato 3.12), [Ollama](https://ollama.com/) con almeno un modello Vision installato e le dipendenze indicate in `requirements.txt` (`rapidfuzz`, `PyYAML`). Le risorse hardware richieste dipendono dal modello.

Organizzare localmente i dati così, usando lo stesso nome di base per immagine e ground truth:

```text
data/
├── screenshot/
│   ├── libretto\\\_001.png
│   └── ...
└── text\\\_visionato/
    ├── libretto\\\_001.txt
    └── ...
```

I principali script sono `run\\\_ocr\\\_models.py` (esecuzione OCR), `normalize\\\_texts.py` e `evaluate\\\_normalized.py` (metriche globali), `field\\\_analysis/parse\\\_fields.py` e `field\\\_analysis/evaluate\\\_fields.py` (analisi per campo). I template si trovano in `field\\\_analysis/templates/`.

**Limite di riproducibilità:** gli esperimenti storici richiedono gli stessi dati, prompt e output descritti nella relazione. Il runner pubblicato usa il prompt definito nella costante `OCR\\\_PROMPT` e **non seleziona automaticamente le tre strategie di prompting storiche**; per riprodurre una configurazione occorre impostare il prompt corrispondente. La procedura separata usata per DeepSeek-OCR e il ricalcolo delle metriche globali POST non sono automatizzati dagli script pubblicati.

## Comandi principali

Installazione:

```bash
git clone https://github.com/alessiopgg/project\\\_librettiEx\\\_ocr.git
cd project\\\_librettiEx\\\_ocr
python -m pip install -r requirements.txt
```

Dopo aver avviato Ollama e preparato `data/`, scaricare un modello ed eseguire l'OCR (il numero dell'esperimento viene assegnato automaticamente):

```bash
ollama pull gemma4:31b
python run\\\_ocr\\\_models.py --models gemma4:31b --num-ctx 8192 --num-predict 4096
```

Normalizzare le trascrizioni e calcolare le metriche globali:

```bash
python normalize\\\_texts.py --clean
python evaluate\\\_normalized.py --clean
```

Eseguire il parsing e la valutazione per campo (qui `experiment\\\_001` è un **percorso di esempio**, da sostituire con quello effettivo):

```bash
python field\\\_analysis/parse\\\_fields.py output/experiment\\\_001/gemma4\\\_31b --output field\\\_analysis/parsed/gemma4\\\_complete
python field\\\_analysis/evaluate\\\_fields.py output/experiment\\\_001/gemma4\\\_31b --dictionary-threshold 70 --dictionary-margin 15 --output field\\\_analysis/results/gemma4\\\_complete
```

Gli output vengono salvati localmente in `output/`, `evaluation/`, `field\\\_analysis/parsed/` e `field\\\_analysis/results/`, cartelle escluse dalla pubblicazione Git. Per tutte le opzioni disponibili, eseguire `python <script>.py --help`.


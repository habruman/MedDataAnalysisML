# MedDataAnalysisML

Dieses Repo wurde im Rahmen meiner Bachelorarbeit an der Phillips Universität Marburg erstellt.

Ziel ist: Anwendung von ML auf drei Arten von nicht-invasiven medizinischen Daten (EKG, Aktigraphie, EMG) zur Klassifizierung verschiedener Arten von Arrhythmien, zur Vorhersage von Depressionsverläufen und zur Klassifizierung von Handgesten mittels Clustering.

## Struktur vom Code im ersten Fall: Fall EKG

- **Konvertierung Rohdaten**
  - `convert.py`: Python-Skript zur Konvertierung der Rohdaten in CSV-Dateien
  - `roh_data`: EKG-Rohdaten in verschiedenen Formaten (.hea, .atr, .xws, .dat)
- **Data**
  - `ekg_data.csv`: Die konvertierten Daten im CSV-Format

- **SVM**
  - `dataset.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien (Standardisierung, Aufteilung)
  - `model.py`: Allgemeine Oberklasse für Klassifikationsalgorithmen
  - `svm_algo.py`: Spezifische Implementierung von SVM
  - `run_svm.py`: Ausführungsskript von SVM

- **DT**
  - `dataset.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien (Standardisierung, Aufteilung)
  - `model.py`: Allgemeine Oberklasse für Klassifikationsalgorithmen
  - `dtree.py`: Spezifische Implementierung von DT
  - `run_dtree.py`: Ausführungsskript von DT

- **KNN**
  - `dataset.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien (Standardisierung, Aufteilung)
  - `model.py`: Allgemeine Oberklasse für Klassifikationsalgorithmen
  - `knn.py`: Spezifische Implementierung von kNN
  - `run_knn.py`: Ausführungsskript von kNN

### Ausführung
- Python-Skript `convert.py` zur Konvertierung der Daten ausführen, dann Ausführungsskript vom gewünschten Algorithmus starten.
- Es wurden relative Pfade verwendet, daher sind alle Skripte direkt startklar.


** Struktur vom Code im zweiten Fall:
Fall Depression
│
├── Data                              -> enthält die Daten bereits im CSV-Format
│   └── activity_data.csv             -> die Daten in CSV-Format
│
├── DeepAR                             
│   └── dataset.py                    -> Python-Skript zum Laden und Vorbereiten der CSV-Dateien 
│   └── deepar.py                     -> spezifische Implementierung von DeepAR
│   └── run_deepar.py                 -> Ausführungsskript von DeepAR
│
├── N-BEATS                       
│   └── dataset.py                    -> Python-Skript zum Laden und Vorbereiten der CSV-Dateien 
│   └── wrapper.py                    -> spezifische Implementierung von N-BEATS
│   └── run_nbeats.py                 -> Ausführungsskript von N-BEATS
│
└── Transformer                      
   └── dataset.py                     -> Python-Skript zum Laden und Vorbereiten der CSV-Dateien
   └── wrapper.py                     -> spezifische Implementierung von Transformer
   └── run_transformer.py             -> Ausführungsskript von Transformer

- Ausführung: Ausführungsskript vom gewünschten Algorithmus starten.
- Auch hier wurden realtive Pfade verwendet, daher sind alle Skripte direkt startklar.


** Struktur vom Code im dritten Fall:
Fall EMG
│
├── Konvertierung Rohdaten            -> enthält die EMG-Rohdaten und Python-Skript zur Konvertierung  -> Daten werden konvertiert und im CSV-Format hier gepeichert
│   └── convert.py                    -> Python-Skript zur Konvertierung der Rohdaten in CSV-Dateien
│
└── Kmeans                      
   └── kmeans.py                      -> Python-Skript zum Laden und Vorbereiten der CSV-Dateien mit spezifischer Implementierung von dem Algorithmus K-means

- Ausführung: Python-Skript convert.py zur Konvertierung der Daten ausführen, dann Python-Skript kmeans.py starten.
- Auch hier wurden realtive Pfade verwendet, daher sind alle Skripte direkt startklar.



















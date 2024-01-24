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

## Struktur vom Code im zweiten Fall: Fall Depression

- **Data**
  - `activity_data.csv`: Die Daten im CSV-Format

- **DeepAR**
  - `dataset.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien
  - `deepar.py`: Spezifische Implementierung von DeepAR
  - `run_deepar.py`: Ausführungsskript von DeepAR

- **N-BEATS**
  - `dataset.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien
  - `wrapper.py`: Spezifische Implementierung von N-BEATS
  - `run_nbeats.py`: Ausführungsskript von N-BEATS

- **Transformer**
  - `dataset.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien
  - `wrapper.py`: Spezifische Implementierung von Transformer
  - `run_transformer.py`: Ausführungsskript von Transformer

### Ausführung
- Direkter Start des Ausführungsskripts vom gewünschten Algorithmus.
- Relative Pfade wurden verwendet, daher sind alle Skripte direkt startklar.


## Struktur vom Code im dritten Fall: Fall EMG

- **Konvertierung Rohdaten**
  - `convert.py`: Python-Skript zur Konvertierung der Rohdaten in CSV-Dateien
  - `roh_data`: EMG-Rohdaten in verschiedenen Formaten

- **Kmeans**
  - `kmeans.py`: Python-Skript zum Laden und Vorbereiten der CSV-Dateien mit spezifischer Implementierung von K-means

### Ausführung
- Python-Skript `convert.py` zur Konvertierung der Daten ausführen, dann `kmeans.py` starten.
- Auch hier wurden relative Pfade verwendet, daher sind alle Skripte direkt startklar.

---

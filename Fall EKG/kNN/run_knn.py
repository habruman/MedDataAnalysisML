import numpy as np 
import dataset as ds
import seaborn as sns
from knn import KNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report



print("Skript gestartet")


dataset_filenames = ds.get_datasets()
print("Gefundene Dateien:", dataset_filenames)

for filename in dataset_filenames:

    print(f"KNN für {filename} ist gestartet")

    # Datensatz laden und verarbeiten
    dataset = ds.Dataset(filename)
    dataset.load_dataset(test_size=0.2)  # oder einen anderen Test-Size-Wert

    
    # KNN-Modell für den aktuellen Datensatz instanziieren
    knn_model = KNN(title=f'KNN für {filename}', dataset=dataset, weights='distance', fast=False)

    knn_model.tune(dataset, verbose=1)
    
    # Modell trainieren und bewerten
    train_time = knn_model.fit(dataset)
    f1_score, run_time, valid_acc, preds = knn_model.score(dataset, train=False)  # False, um auf Testdaten zu bewerten

    print(f"Trainingszeit: {train_time}s, F1-Score: {f1_score}, Genauigkeit der Vorhersagen: {valid_acc}, Laufzeit: {run_time}s")

    if knn_model.is_trained:
        
        # Konfusionsmatrix für Leistung des Klassifikationsmodells
        cm = confusion_matrix(dataset.y_test, preds)
        labels = np.unique(np.concatenate((dataset.y_test, preds)))
        cm_percent = cm/np.sum(cm)
        sns.heatmap(cm_percent, annot=True, fmt=".3%", cmap='Reds', xticklabels=labels, yticklabels=labels)
        plt.title(f'Konfusionsmatrix der Aufnahme {dataset.filename}')
        plt.ylabel('Tatsächliche Klasse')
        plt.xlabel('Vorhergesagte Klasse')
        plt.show()
        
    
    
    
    
    
    
        # Klassifikationsbericht von Scikit-learn
        print(classification_report(dataset.y_test, preds, zero_division=1))
        
        # Um Warnungen zu behandeln, wird zero_division auf 1 gesetzt. So wird die Metrik auf 1 gesetzt, wenn eine Division durch Null auftritt
        
        
        
    
        # Matthews-Korrelationskoeffizient (MCC), von -1 (vollständige Unstimmigkeit) bis +1 (perfekte Vorhersage)
        mcc = matthews_corrcoef(dataset.y_test, preds)
        print(f"MCC: {mcc}")
    
    print(f"KNN für {filename} ist beendet")
    
    print("=" * 100)


print("Skript beendet")    
    
    

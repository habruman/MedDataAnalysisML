import numpy as np 
import dataset as ds
import seaborn as sns
from svm_algo import SVM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report



print("Skript gestartet")


dataset_filenames = ds.get_datasets()
print("Gefundene Dateien:", dataset_filenames)

for filename in dataset_filenames:
    
    print(f"SVM für {filename} ist gestartet")

    # Datensatz laden und verarbeiten
    dataset = ds.Dataset(filename)
    dataset.load_dataset(test_size=0.2)  # oder einen anderen Test-Size-Wert
    
    
    # SVM-Modell für den aktuellen Datensatz definieren
    svm_model = SVM(title=f"SVM für {dataset.filename}", dataset=dataset, kernel='rbf', fast=False)

    svm_model.tune(dataset, verbose=1)
    
    # Modell trainieren und bewerten
    train_time = svm_model.fit(dataset)
    f1_score, run_time, valid_acc, preds = svm_model.score(dataset, train=False)  # False, um auf Testdaten zu bewerten

    print(f"Trainingszeit: {train_time}s, F1-Score: {f1_score}, Genauigkeit der Vorhersagen: {valid_acc}, Laufzeit: {run_time}s")
    
    if svm_model.is_trained:
        
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
    
        # Matthews-Korrelationskoeffizient (MCC), von -1 (vollständige Unstimmigkeit) bis +1 (perfekte Vorhersage)
        mcc = matthews_corrcoef(dataset.y_test, preds)
        print(f"MCC: {mcc}")
    
    print(f"SVM für {filename} ist beendet")
    
    print("=" * 100)


print("Skript beendet")
import os
import csv
import wfdb
import numpy as np 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

print("Konvertierung geht los")


# Pfad zum Verzeichnis mit den EKG-Aufzeichnungen
directory_path = Path(".")

# Liste der Dateien (files) im Verzeichnis, die mit dat enden
file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.dat')]


# gehe durch jede Datei und führe das Konvertierungsskript aus
for file_name in file_names:
    # Extrahiere den Basisname ohne Dateiendung. Null um das erste Element aus der Ergebnisliste zu nehmen
    base_name = file_name.split('.')[0]
    # Pfad zu der Datei machen
    record_path = os.path.join(directory_path, base_name)
    print(f"Konvertierung der Aufnahme {base_name}:")

    # Lese die Signaldaten aus dat und Header aus hea mit rdsamp Funktion von wfdb
    signals, fields = wfdb.rdsamp(record_path)

    # ein DataFrame für die Signaldaten. sig_name ist Schlüssel von rdsamp für die extrahierte Namen 
    df_signals = pd.DataFrame(signals, columns=fields['sig_name'])


    # die Annotation extrahieren, die in den atr Dateien sind. Das macht die Funktion rdann
    annotation = wfdb.rdann(record_path, 'atr')

    # ein DataFrame für die Annotation
    # Annahme, dass Sample und Symbol vorhanden sind, durch Speicherung der gesamten atr Datei
    df_annotations = pd.DataFrame({'Sample': annotation.sample,'Symbol': annotation.symbol})

    
    # jetzt Daten in eine Liste holen
    # zip damit beiden Spalten Sample und Symbol paarweise zusammenzuführen
    extracted_data = list(zip(df_annotations['Sample'], df_annotations['Symbol']))   
 
    # segmente und Herzschlagtypen
    half_second_size = 180 # Hälfte von dem Segment
    atr_types = ['N','A','V','F','E','O','P','L','f','Q','/','|','!','R']
    types_counter = [0]*len(atr_types)

    segments = []     # für alle segmente
    label_index = []     # für atr_type Index
    segment = [] # für ein einziges segment mit 360 signal

    # segmentieren
    for idx in range(len(annotation.sample)):
        pos = annotation.sample[idx] # Position von Sample
        atr_type = annotation.symbol[idx] # Typ von der Arrhythmie

        if atr_type in atr_types:
            atr_index = atr_types.index(atr_type)
            types_counter[atr_index] += 1
            # pos muss größer 180 und kleiner (länge der signalliste - 180)
            # so wird gesichert, dass es vor und nach pos genug signale gibt -> dann sind alle segmente gleich lang
            if half_second_size <= pos < (len(signals) - half_second_size):
                segment = signals[pos-half_second_size:pos+half_second_size, 0] # die erste Spalte von signals, das ist mlii oder in einem einzigen Fall eine andere Ableitung
                segments.append(segment)
                label_index.append(atr_index)

  
        
    print("Dimension von der Liste mit den Messungen") 
    print(np.shape(segments))
    print("Dimension von der Liste mit den Labeln") 
    print(np.shape(label_index))
        
  
    # Label-Index zu jedem Segment hinzufügen
    for i in range(0,len(segments)):
            segments[i] = np.append (segments[i],label_index[i] )
    print("Dimension von der Liste mit den Messungen nach Hinzufügen von der Spalte der Labeln") 
    print(np.shape(segments))

    # Daten in csv speichern
    with open(f"../Data/{base_name}_X_data_with_label.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for value in segments:
            writer.writerow(value)


    label_index_df = pd.DataFrame(label_index) # Liste label_index in Dataframe konvertieren, damit value_counts() verwendet werden kann
    class_num = label_index_df[0].value_counts() # value_counts() gibt die Anzahl der Arrhythmie Typen in der Spalte
    print("Die Vorkommen der verschiedenen Arrhythmie Typen")
    print(class_num)
    
    

 
 
    
    # Berechnung von Prozentsätzen für die Visualisierung
    percentages = [value / sum(class_num) * 100 for value in class_num]
    
    # Legenden-Labels mit den Prozentsätzen
    legend_labels = [f'{label} - {percentage:.1f}%' for label, percentage in zip(atr_types, percentages)]


    # kuchendiagramms :)
    plt.figure(figsize=(12,5))
    plt.pie(class_num, colors=['tab:red','tab:orange','tab:green','tab:olive','tab:blue'])
    plt.title(f"Anteile der extrahierten Arrhythmie Typen in Aufnahme {base_name}")
    plt.axis('equal')  
    plt.legend(legend_labels, loc="best", bbox_to_anchor=(1, 1)) 
    plt.show()

    

    
    print("=" * 100)
    
        
print("Konvertierung abgeschlossen, Glückwunsch!")


 
    
    
        
        
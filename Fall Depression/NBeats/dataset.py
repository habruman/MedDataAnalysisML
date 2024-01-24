from os import walk
import pandas as pd
from pathlib import Path



print("Skript gestartet")

def get_datasets():
    
    # Verwenden des aktuellen Verzeichnisses für den Datenpfad

    data_folder = "../Data" 
    filenames = next(walk(data_folder), (None, None, []))[2] # durchsucht die gesamte Ordnerstruktur und nimmt alle Dateinamen 
    filenames = list(filter(lambda x: 'csv' in x, filenames)) # filtert die Namen und nimmt nur die Dateien, die CSV im Namen enthalten und dann wird der iterator in eine noramle Liste umgewandelt
    return filenames

print(f"Holen von den Datensätzen ist fertig") 



def load_and_prepare_data():
    
    dataset_filenames = get_datasets()
    dataframes = []
    
    for filename in dataset_filenames:
        data_file = Path("../Data") / filename
        df = pd.read_csv(data_file)
        
        # Konvertierung der Zeitstempel Spalte in ein DateTime-Objekt und sie als Index setzen
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = pd.DataFrame({'timestamp': df['timestamp'], 'target': df['activity']})
        df.set_index('timestamp', inplace=True)

        dataframes.append((filename, df))
        print(f"Datensatz: {filename} ist geladen und auch vorbereitet")
        
        
    return dataframes
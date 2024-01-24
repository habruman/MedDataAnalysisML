import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def convert(trennzeichen='\t', time='time'):
    ordnerpfad = Path(".")
    for root, dirs, files in os.walk(ordnerpfad):
        for file in files:
            if file.endswith('.txt'):
                # Pfad der aktuellen Datei
                dateipfad = os.path.join(root, file)

                # Einlesen der Textdatei
                df = pd.read_csv(dateipfad, delimiter=trennzeichen)

                
                
                if time in df.columns:
                    df[time] = pd.to_datetime(df[time], unit='ms')

                


                # Auswahl der Spalten für die Standardisierung (channel1 bis channel8)
                channel_features = ['channel1', 'channel2', 'channel3', 'channel4', 
                                    'channel5', 'channel6', 'channel7', 'channel8']
                
                # Überprüfen, ob die channel_features vorhanden sind
                if all(feature in df.columns for feature in channel_features):
                    # Standardisierung der EMG-Kanäle
                    scaler = StandardScaler()
                    df[channel_features] = scaler.fit_transform(df[channel_features])
                else:
                    print(f"Einige der benötigten Channel-Features fehlen in {file}.")
                

                # Erstellen des neuen Dateipfads für die CSV-Datei
                csv_dateipfad = os.path.splitext(dateipfad)[0] + '.csv'

                # Speichern als CSV
                df.to_csv(csv_dateipfad, index=False)
                print(f"Konvertiert und standardisiert: {dateipfad} zu {csv_dateipfad}")

convert()

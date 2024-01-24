from os import walk
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def get_datasets():
    

    # Verwenden des aktuellen Verzeichnisses für den Datenpfad

    data_folder = "../Data" # # im Verzeichnis Data
    filenames = next(walk(data_folder), (None, None, []))[2] # durchsucht die gesamte Ordnerstruktur und nimmt alle Dateinamen 
    filenames = list(filter(lambda x: 'csv' in x, filenames)) # filtert die Namen und nimmt nur die Dateien, die CSV im Namen enthalten und dann wird der iterator in eine noramle Liste umgewandelt
    return filenames


print("Holen von Datensätzen ist fertig") 

    
class Dataset:
    # Konstruktor
    def __init__(self, filename: str):
        self.name = filename[:-4]  # entfernt 4 Zeichen. Angenommen, die Dateiendung ist .csv/  und speichert das Ergebnis als den Namen des Datensatzes.
        self.filename = filename
        # attribute der Klasse
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.class_num = None
        self.feature_count = None
                
        
    def load_dataset(self, test_size=0.2): #Anteil der Testdaten 20%

        data_file = Path("../Data") / self.filename # sucht nur die Datei mit Filename in Ordner aus
        df = pd.read_csv(data_file, header=None) # header=None sehr wichtig, weil wir keine Kopfzeile haben, sonst wird die erste Datenzeile als Kopf genommen
        
        print(f"Load Datensatz: {self.name}") 

        # Aufteilung der Daten in Features und Labels
        # The last column is the labels for both datasets
        self.x = df.iloc[:, :-1]  # alle Spalten außer der letzten werden als Features (x) verwendet
        self.y = df.iloc[:, -1]   # die letzte Spalte wird als Label (y) verwendet

        self.split_dataset(self.x, self.y, test_size)

        
            # Standardisierung der Daten
            # fit berechnet den Mittelwert und die Standardabweichung der Daten nur im Trainingsset und skaliert sie dann mit transform
            # transform alleine standardisiert die Testdaten anhand der berechneten werten von Trainingssets, um die gleiche Vorverarbeitung  zu sichern
            # so werden keine Infos von den Testdaten in die Trainigsdaten gemischt was eine genauere Arbeit ermöglicht
        standard_scaler = StandardScaler()
        self.x_train = standard_scaler.fit_transform(self.x_train)
        self.x_test = standard_scaler.transform(self.x_test)

        self.class_num = np.unique(self.y) # findet alle einzigartigen Werte im Array self.y heraus
        self.feature_count = self.x.shape[1] #gibt die Anzahl der Spalten in self.x an, also Anzahl der Features
        print("Die Indizes der Arrhythmie-Typen im Datensatz:")
        print(self.class_num)
        print("Anzahl der Feature im Datensatz ist:")
        print(self.feature_count)


              
        print("Die Art der Klassifizierung im Datensatz ist:")
        if self.class_num.size == 2:
            print("binary")
        elif self.class_num.size < 100:
            print("multi-label")
        
    # random_state, um Zufälligkeit zu steuern / 42 hat keine Bedeutung, hauptsache eine feste Zahl
    def split_dataset(self, x, y, test_size=0.20, random_state=0):

    # train_test_split ist in scikit-learn vordefiniert
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,  y, test_size=test_size, random_state=random_state)
        
        print(f"Split Datensatz {self.name} ist fertig")    
        
        print("X_train : ", np.shape(self.x_train))
        print("X_test : ", np.shape(self.x_test))
        print("y_train : ", np.shape(self.y_train))
        print("y_test : ", np.shape(self.y_test))


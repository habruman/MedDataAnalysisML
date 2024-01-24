import pickle
import timeit
import numpy as np 
import pandas as pd
from pathlib import Path
from dataset import Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

class Model:

    def __init__(self, title: str, model, dataset: Dataset, fast: bool):
        self.title = title
        self.model = model
        self.hyper_param_distribution = None
        self.fast = fast # schnelle Version der Modelltraining oder Bewertung
        self.scoring_metric = 'f1_macro' # Metrik zur Bewertung des Modells
        self.feature_count = dataset.feature_count
        self.class_num = dataset.class_num
        self.is_trained = False
        # wie viele verschiedene Hyperparameter-Kombis sollen während der Optimierung ausprobiert werden?
        self.n_iter_search = 10
        if self.fast:
            self.n_iter_search = 5

    # trainiere das Modell mit dem Datensatz
    def fit(self, dataset: Dataset):

        start = timeit.default_timer()
        
        unique_classes = np.unique(dataset.y_train)
        
        if len(unique_classes) > 1:
        # die Methode fit des gegebenen Modells
            self.model.fit(dataset.x_train, dataset.y_train)
            self.is_trained = True
            stop = timeit.default_timer()
            train_time = round(stop - start, 3) # Das Ergebnis wird auf drei Dezimalstellen gerundet
        else:
            print("Fehler: Nur eine Klasse im Set vorhanden.") 
            train_time = 0
        return train_time

    # bewerte die Leistung des Modells und messe dabei die Zeit
    def score(self, dataset: Dataset, train):

        start = timeit.default_timer()

        # über train wird entschieden, ob das Modell anhand der Trainingsdaten (True) oder der Testdaten (False) bewertet werden soll
        if train:
            x = dataset.x_train
            y = dataset.y_train
            mode = 'train'
        else:
            x = dataset.x_test
            y = dataset.y_test
            mode = 'test'
        if self.is_trained:    
        # mache Vorhersagen
            preds = self.model.predict(x)
            print("Vorhersagen:", preds)
        
        # mache x von npArray zu Dataframe und füge die Vorhersagen zu den Rohdaten hinzu und speichere dann alles in csv Datei
            df = pd.DataFrame(x)
        
            df['Vorhersagen'] = preds
           

        
            current_dir = Path('.')
            set_name = dataset.name[:3]
            models_dir = current_dir / 'Vorhersagen' 
            models_dir.mkdir(parents=True, exist_ok=True)  # Erstellen des Ordners, falls das nicht 
            model_filepath = models_dir / ("Vorhersagen für Aufnahme "+ set_name + ".csv")

            df.to_csv(model_filepath, index=False, header=None)
        
        
        
        # Laufzeit für die Erstellung der Vorhersagen
            stop = timeit.default_timer()
            run_time = round(stop - start, 3)

        # y_true sind wahren Labels und y_pred die vom Modell vorhergesagten Labels
        # Berechnet die Genauigkeit der Vorhersagen
            valid_acc = accuracy_score(y_pred=preds, y_true=y)
        # Berechnet den F1-Score der Vorhersagen
            f1 = f1_score(y, preds, average='macro')

        #print(valid_acc)
        else:
            print(f"Das Modell wurde für {dataset.name} nicht trainiert. Überspringe die Bewertung.")
            f1 = None
            run_time = None
            valid_acc = None 
            preds = None
        return f1, run_time, valid_acc, preds

    def save(self, dataset_name: str):

        # Erstellen des Pfades für den modells-Ordner im aktuellen Verzeichnis
        current_dir = Path('.')
        set_name = dataset_name[:3]
        models_dir = current_dir / 'Modelle'
        models_dir.mkdir(parents=True, exist_ok=True)  # Erstellen des Ordners, falls das nicht existiert

        # Erstellen des Pfades für die Modell-Datei
        model_filepath = models_dir / ("Modell von der Aufnahme " + set_name + ".pkl")
        
        # Speichern des Modells / wb = "write binary"
        pickle.dump(self.model, open(model_filepath, 'wb'))
    
    
    def load(self, dataset_name: str):
  
        # Erstellen des Pfades für den modells-Ordner im aktuellen Verzeichnis
        current_dir = Path('.')
        set_name = dataset_name[:3]
        models_dir = current_dir / 'Modelle' 

        # Erstellen des Pfades für die Modell-Datei
        model_filepath = models_dir / ("Modell von der Aufnahme " + set_name + ".pkl")

        # Laden des Modells / rb = "read binary"
        self.model = pickle.load(open(model_filepath, 'rb'))

        
        # Hyperparameteroptimierung
    def tune(self, dataset: Dataset, verbose=0):
        unique_classes = np.unique(dataset.y_train)
        if len(unique_classes) > 1:
            start = timeit.default_timer()
            n_jobs = -2  # nutze alle verfügbaren Prozessoren bis auf zwei, um die Suche zu beschleunigen
        
        # RandomizedSearchCV führt eine zufällige Suche über Hyperparametergruppen durch
        # scoring_metric bestimmt die Metrik, die zur Bewertung der Modellleistung während der Suche verwendet wird
        # verbose steuert die Ausgabe während der Suche / 0 bedeutet, dass keine Ausgabe erfolgt / 1 oder 2 geben mehr Informationen z.B. wie viele Kombinationen bereits getestet wurden
            clf = RandomizedSearchCV(self.model, self.hyper_param_distribution, random_state=0, scoring=self.scoring_metric,
                                 n_iter=self.n_iter_search, n_jobs=n_jobs, verbose=verbose)
            search = clf.fit(dataset.x_train, dataset.y_train)

        # cv_results_ ist vordefiniert von RandomizedSearchCV
        # speichert Ergebnisse jedes Durchlaufs, um das beste modell zu finden
        # das beste Modell ist das mit der niedrigsten rank_test_score
            self.search_results = search.cv_results_

            stop = timeit.default_timer()
       
        # Speichere das beste Modell
            self.model = search.best_estimator_
            self.save(dataset.name)
        
            self.save_search_results(dataset.name)
        else:
            print(f"Das Modell wurde für {dataset.name} nicht getunt, da nur eine Klasse vorhanden ist.")
          
        # konvertiert die Ergebnisse von tune in ein DataFrame und speichert sie dann in einer csv Datei
    def save_search_results(self, dataset_name: str):
        
        current_dir = Path('.')
        set_name = dataset_name[:3]
        models_dir = current_dir / 'Ergebnis von Tune' 
        models_dir.mkdir(parents=True, exist_ok=True)  # Erstellen des Ordners, falls das nicht existiert
        
        model_filepath = models_dir / ("Ergebnis von Tune für die Aufnahme "+ set_name + ".csv")
        
        results_df = pd.DataFrame(self.search_results)
        results_df.to_csv(model_filepath, index=False)      
       
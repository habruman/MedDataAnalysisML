import pickle
import numpy as np
from pathlib import Path
from seedpy import fixedseed
from gluonts.mx import Trainer
from gluonts.mx import DeepAREstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import OffsetSplitter



class DeepARWrapper:
    
    # n_epochs: Anzahl der Runden, für die der DeepAR-Algorithmus trainiert werden soll
    # lag gibt an, wie viele vorherige Zeitpunkte in der Zeitreihe berücksichtigt werden, um den zukünftigen Wert vorherzusagen
    def __init__(self, n_epochs, lag):
        self.n_epochs = n_epochs
        self.lag = lag
        self.is_fitted = False

    # Modelltraining
    # T: Frequenz der Zeitreihe 'minutlich'
    # target: Zielvariable
    def fit(self, X):
        
        test_size = int(0.25 * len(X))
        
        dataset = PandasDataset(X, freq='T', target="target")
        
        # die letzten (test_size) Werte für das Testset verwenden
        splitter = OffsetSplitter(offset=-test_size)
        
        # Trainingsdaten für das Modelltraining, Testdaten werden hier mit '_' ignoriert
        training_data, _ = splitter.split(dataset)
        
        self.model = DeepAREstimator(prediction_length=1, freq="T", trainer=Trainer(epochs=self.n_epochs)).train(training_data)
        self.is_fitted = True

    # Vorhersageerstellung
    # 
    def predict(self, X):

        test_size = int(0.25 * len(X))
        
        dataset = PandasDataset(X,freq='T', target="target")

        # die letzten 'test_size' Werte für das Testset verwenden
        splitter = OffsetSplitter(offset=-test_size)
        
        # Testdaten für die Vorhersage, Trainingsdaten werden hier mit '_' ignoriert
        _, test_gen = splitter.split(dataset)
        
        # Instanzen aus Testdaten für die Methode predict erstellen
        # die Methode predict braucht diese Instanzen als Eingabe
        # windows: Anzahl der Fenster für die Vorhersage ist gleich test_size, das heißt es werden genau soviele Fenster sein wie Anzahl der Daten in Testset
        # distance: zwischen den Fenstern liegt eine Distanz von 1 Datenpunkt 
        test_data = test_gen.generate_instances(prediction_length=1, windows=test_size, distance=1)

        forecasts = list(self.model.predict(test_data.input))
        
        # DeepAR generiert für jede Vorhersage mehrere Stichproben (Samples) 
        # mean: berechnet den Mittelwert dieser Stichproben 
        # die berechneten Mittelwerte sind dann die Vorhersagen
        predictions = np.array([x.samples.mean() for x in forecasts]).squeeze()
        
        # die ersten (self.lag) Vorhersagen auf die echten Werte der Zeitreihe setzen
        predictions[:self.lag] = X['target'][-test_size:-test_size+self.lag]
        
        return predictions

    def save(self, dataset_name: str):

        # Pfad für den modells-Ordner im aktuellen Verzeichnis
        current_dir = Path('.')
        models_dir = current_dir / 'models' / dataset_name
        models_dir.mkdir(parents=True, exist_ok=True)  # Erstellen des Ordners, falls das nicht existiert

        #  Pfades für die Modell-Datei
        model_filepath = models_dir / (dataset_name[:-4] + ".pkl")

        # Speichern des Modells / wb = "write binary"
        pickle.dump(self.model, open(model_filepath, 'wb'))
        

    def load(self, path):
        from gluonts.model.predictor import Predictor
        # deserialize aus GluonTS. liest die gespeicherte Datei und stellt das Modellwieder herr
        self.model = Predictor.deserialize(Path(path))



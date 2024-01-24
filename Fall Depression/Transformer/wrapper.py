import pickle
import numpy as np
from pathlib import Path
from seedpy import fixedseed
from gluonts.mx import Trainer
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import OffsetSplitter

class GluonTSWrapper:

    def __init__(self, model, n_epochs, lag):
        self.model = model
        self.n_epochs = n_epochs
        self.lag = lag
        self.is_fitted = False

    def fit(self, X):

        test_size = int(0.25 * len(X))
        
        dataset = PandasDataset(X, freq='T', target="target")

        splitter = OffsetSplitter(offset=-test_size)
        training_data, _ = splitter.split(dataset)
        self.model = self.model(prediction_length=1, freq="T", trainer=Trainer(epochs=self.n_epochs)).train(training_data)
        self.is_fitted = True

        
    def predict(self, X):

        test_size = int(0.25 * len(X))

        dataset = PandasDataset(X, freq='T', target="target")

        splitter = OffsetSplitter(offset=-test_size)
        _, test_gen = splitter.split(dataset)

        test_data = test_gen.generate_instances(prediction_length=1, windows=test_size, distance=1)

        forecasts = list(self.model.predict(test_data.input))
        predictions = np.array([x.samples.mean() for x in forecasts]).squeeze()

        
        predictions[:self.lag] = X['target'][-test_size:-test_size+self.lag]
        
        return predictions

    def save(self, dataset_name: str):

        # Pfad für den modellOrdner im aktuellen Verzeichnis
        current_dir = Path('.')
        models_dir = current_dir / 'models' / dataset_name
        models_dir.mkdir(parents=True, exist_ok=True) 

        # Erstellen des Pfades für die Modell-Datei
        model_filepath = models_dir / (dataset_name[:-4] + ".pkl")

        # Speichern des Modells / wb = "write binary"
        pickle.dump(self.model, open(model_filepath, 'wb'))
        

    def load(self, path):
        from gluonts.model.predictor import Predictor
        self.model = Predictor.deserialize(Path(path))



import dataset
import numpy as np
from seedpy import fixedseed
import matplotlib.pyplot as plt
from deepar import DeepARWrapper

# parameter definieren für DeepARWrapper
# model_init_seed für Reproduzierbarkeit, wenn daten zufällig generiert werden 
# lag ist die Zeitschritte in Vergangenheit 
hyperparameters = {
    'deepar': {
        'n_epochs': 1,  
        'model_init_seed': 198471,
        'lag': 5,
    }
}
# holen von allen csv-Dateien im Verzeichnis
dataset_filenames = dataset.get_datasets()
print("Gefundene Dateien:", dataset_filenames)

# laden und vorverarbeiten von allen csv-Dateien im Verzeichnis    
dataframes = dataset.load_and_prepare_data()

for filename, X in dataframes:
    
        print("Verarbeite Datei:", filename)

            # deepar aufrufen
        hyp = hyperparameters['deepar']
        deepar = DeepARWrapper(n_epochs=hyp['n_epochs'], lag=hyp['lag'])
        with fixedseed(np, seed=hyp['model_init_seed']):
                    deepar.fit(X)
                    deepar.save(filename)
           
        preds = deepar.predict(X)
             
 
        test_size = int(0.25 * len(X))
        y_test = X['target'][-test_size:]
        
        X['predictions'] = 0
        
        X.iloc[-len(preds):, X.columns.get_loc('predictions')] = preds


        # Speichern in eine CSV-Datei
        X.to_csv("Vorhersagen für Datei "+ filename, index=True)

        print(preds)
        
        
    

        # Visualisierung
        plt.figure(figsize=(10, 6))

        plt.plot(X.index[-test_size:], y_test, label='Tatsächliche Werte', color='blue')

        plt.plot(X.index[-test_size:], preds, label='Vorhersagen', color='red')

        plt.title('Vergleich von tatsächlichen Werten und Vorhersagen für Datei ' + filename)
        plt.xlabel('Zeit')
        plt.ylabel('Wert')
        plt.legend()

        plt.show()
        
        
    
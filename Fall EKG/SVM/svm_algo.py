import numpy as np
from Model import Model
from sklearn.svm import SVC
from dataset import Dataset


class SVM(Model):
    
    
    def __init__(self, title: str, dataset: Dataset, kernel: str, fast: bool):
        # SVC: Support Vector Classifier ist aus sklearn.svm
        super().__init__(title, SVC, dataset, fast)
        self.kernel = kernel #  bestimmt, wie Daten im Modell verarbeitet und dargestellt werden
        self.max_iter = 40000 # die maximale Anzahl von Iterationen für den Trainingsprozess

        
        # C: das Ausmaß, in dem das Modell falsche Klassifikationen auf den Trainingsdaten vermeiden soll / großer Wert für C bedeutet, dass das Modell versuchen wird, möglichst viele Trainingsdaten korrekt zu klassifizieren
        # gamma: bestimmt den Radius der rbf Kernel / niedriger Wert führt zu größeren Ähnlichkeitsbereichen, was eine glatter Entscheidungsgrenze macht
        #gamma = np.logspace(-9, 3, 13) # 13 Zahlen logarithmisch erzeugen: mit Startpunkt und Endpunkt
        #C = logspace(-2, 10, 13)

        self.hyper_param_distribution = {'C': np.logspace(-2, 10, 13),'gamma': np.logspace(-9, 3, 13),'max_iter': [self.max_iter]} 
    
        self.model = self.model(kernel=kernel, max_iter=self.max_iter, class_weight='balanced', tol=0.01, random_state=0) 
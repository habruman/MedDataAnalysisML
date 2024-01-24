import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Model import Model
from dataset import Dataset

class KNN(Model):


    def __init__(self, title: str, dataset: Dataset, weights: str, fast: bool):

        super().__init__(title, KNeighborsClassifier, dataset, fast)


            
            
        # uniform: alle Nachbarn haben gleiches Gewicht
        # distance: nähere Nachbarn haben größeren Einfluss  
        self.weights = weights

        
        # list(range(1, 30, 2)) Liste von Zahlen von 1 bis 30, wobei nur jede zweite Zahl aufgenommen wird. Also betrachte nur ungerade Anzahl von Nachbarn für KNN. Damit es nicht zu einer Gleichheit der Klassen kommt 
        self.hyper_param_distribution = {"n_neighbors": list(range(1, 30, 2)),
                                         "weights": ['uniform', 'distance'], "metric": ['euclidean', 'manhattan', 'minkowski']}

        # n_neighbors bestimmt die Anzahl der Nachbarn, die für die Klassifizierung eines Punktes
        self.model = self.model(n_neighbors=3, weights=self.weights)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dataset import Dataset
from Model import Model

class DecisionTree(Model):


    def __init__(self, title: str, dataset: Dataset, fast: bool):
        super().__init__(title, DecisionTreeClassifier, dataset, fast)

        self.hyper_param_distribution = {'max_depth': np.arange(3, 25), 'min_weight_fraction_leaf': np.linspace(0.0, 0.5, 5), 'min_samples_split': np.arange(2, 10), 'min_samples_leaf': np.arange(1, 10), 'class_weight': [None, 'balanced']}


        self.model = self.model(max_depth=20, random_state=0)

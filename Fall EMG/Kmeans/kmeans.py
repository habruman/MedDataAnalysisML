import os
import numpy as np
import pandas as pd
import seaborn as sns 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score


def kmeans():
    hauptordner = '../Konvertierung Rohdaten'
    for ordner in os.listdir(hauptordner):
        ordnerpfad = os.path.join(hauptordner, ordner)
        if os.path.isdir(ordnerpfad):
            csv_dateien = [f for f in os.listdir(ordnerpfad) if f.endswith('.csv')]
            if len(csv_dateien) == 2:
                df1 = pd.read_csv(os.path.join(ordnerpfad, csv_dateien[0]))
                df2 = pd.read_csv(os.path.join(ordnerpfad, csv_dateien[1]))

                
                # Erstellung Liniendiagramme, um Zeitreihendaten von der ersten Serie zu visualisieren
                # Stelle sicher, dass der Index ein Datetime-Index ist
                df1.index = pd.to_datetime(df1.index)  

                # für jeden Kanal ein Diagramm
                for k in range(1, 9):
                    channel_data = df1[f'channel{k}']
                    # Liniendiagramm
                    plt.figure(figsize=(10, 4))
                    plt.plot(channel_data)
                    plt.title(f'EMG Daten - Channel {k}')
                    plt.xlabel('Zeit')
                    plt.ylabel('Messwert')
                    plt.show()



                data1 = df1[df1.columns[1:-1]]
                data2 = df2[df2.columns[1:-1]]
                print(data1.head())
                print(data2.head())

                print(data1.describe()) 
                print(data2.describe()) 
                # deskriptive Statistiken der standardisierten EMG-Daten 
                # um sicherzustellen, dass alle Features (Kanäle) gleich behandelt werden


                # Korrelationsmatrix für Korrelation zwischen den Kanälen der ersten Serie
                corr = data1.corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                cax=ax.matshow(corr,vmin=-1,vmax=1)
                ax.matshow(corr)
                plt.xticks(range(len(corr.columns)), corr.columns)
                plt.yticks(range(len(corr.columns)), corr.columns)
                plt.xticks(rotation=90)
                plt.colorbar(cax)
                plt.title('Korrelationsmatrix')
                plt.show()
    


                # Angenommen, X sind bereits vorbereitete und skalierte Daten
                X1 = data1  # Die Features ohne die Spalte 'class'
                X2 = data2

                
                # Kreuzvalidierung vorbereiten
                kf = KFold(n_splits=7, shuffle=True, random_state=0)
                silhouette_scores_for_k = {k: [] for k in [4, 5, 6, 7, 8, 9, 10]}

                for k in [4, 5, 6, 7, 8, 9, 10]:
                    for train_index, test_index in kf.split(X1):
                        # Teile die Daten in Trainings- und Testsets auf
                        X_train, X_test = X1.iloc[train_index], X1.iloc[test_index]
        
                        # Initialisiere KMeans mit der aktuellen Anzahl von Clustern
                        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        
                        # Passe das Modell an die Trainingsdaten an und erstelle Vorhersagen für das Testset
                        kmeans.fit(X_train)
                        cluster_labels = kmeans.predict(X_test)
        
                        # Berechne den Silhouetten-Score für das aktuelle Testset
                        score = silhouette_score(X_test, cluster_labels)
        
                        # Speichere den Score
                        silhouette_scores_for_k[k].append(score)

                # Ergebnisse ausgeben
                for k, scores in silhouette_scores_for_k.items():
                    print(f"Silhouetten-Scores für k={k}: {scores}")
                    print(f"Durchschnittlicher Silhouetten-Score für k={k}: {np.mean(scores)}\n")
    
                # Berechne die durchschnittlichen Silhouetten-Scores für jedes k
                average_silhouette_scores = [np.mean(scores) for scores in silhouette_scores_for_k.values()]

                # Erstelle eine Liste der k-Werte
                k_values = list(silhouette_scores_for_k.keys())   


                plt.figure(figsize=(8, 6))
                plt.plot(k_values, average_silhouette_scores, '-o')
                plt.xlabel('Anzahl der Cluster k')
                plt.ylabel('Durchschnittlicher Silhouettenwert')
                plt.title('Silhouettenanalyse für verschiedene Werte von k')
                plt.xticks(k_values)  # Stelle sicher, dass alle k-Werte als Ticks angezeigt werden
                plt.show()

                


                # Erstelle ein NumPy-Array aus den Daten, besser für die Methoden fit, predict und pca
                # so bekommen alle das gleiche Array und dann gibt es keine Namenkonflikte der Spalten
                data_array1 = data1.values
                data_array2 = data2.values

                # Führe KMeans mit dem besten ausgewählten k aus, um die Cluster zu bilden
                optimal_k = 7   # Diesen Wert anpassen basierend auf deiner Analyse
                # Returns: trainiertes Modell / https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
                cluster = KMeans(n_clusters=optimal_k, n_init=10, random_state=0).fit(data_array1)

                # Returns: labels, Index von cluster für jeden Wert
                clusters = cluster.predict(data_array2)
                # Speichere die Clusterzuweisungen im orginalen DataFrame
                df2['cluster'] = clusters  




                print(df1.head())
                print(df2.head())

                print(df2['cluster'].value_counts())



                output_dateipfad = 'D:/test5/Clustering_Ergebnis.csv'

                df2.to_csv(output_dateipfad, index=False)



                # Visualisiere die Anzahl der Datenpunkte pro Cluster in einem Barplot

                d = pd.DataFrame(df2['cluster'].value_counts())
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.bar(d.index, d['cluster'], align='center', alpha=0.5)
                plt.xlabel('Cluster')
                plt.ylabel('Anzahl der Datenpunkte')
                plt.title('Anzahl der Datenpunkte pro Cluster')
                plt.show()






                cross_tab = pd.crosstab(df2['class'], df2['cluster'])

                print(cross_tab)


                plt.figure(figsize=(10, 8))
                sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Vorhergesagte Cluster')
                plt.ylabel('Tatsächliche Handgeste')
                plt.title('Heatmap der Kreuztabelle')
                plt.show()







                # PCA für 2 Komponenten 
                pca = PCA(n_components=2)
                data_pca = pca.fit_transform(data_array2)
                cluster_centers_pca = pca.transform(cluster.cluster_centers_)


                plt.figure(figsize=(14, 7))

                # Plot für KMeans-Cluster
                plt.subplot(1, 2, 1)
                sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=df2['cluster'], palette='viridis', s=50, alpha=0.6)
                plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], s=20, c='red', marker='x')
                plt.title('2D-Visualisierung der KMeans-Cluster')
                plt.xlabel('Erste Hauptkomponente')
                plt.ylabel('Zweite Hauptkomponente')

                # Plot für tatsächliche Geste
                plt.subplot(1, 2, 2)
                sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=df2['class'], palette='Set2', s=50, alpha=0.6)
                plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], s=20, c='red', marker='x')
                plt.title('2D-Visualisierung der tatsächlichen Handgesten')
                plt.xlabel('Erste Hauptkomponente')
                plt.ylabel('Zweite Hauptkomponente')

                plt.tight_layout()
                plt.show()




                # PCA für 3 Komponenten
                pca_3d = PCA(n_components=3)
                data_pca_3d = pca_3d.fit_transform(data_array2)

                fig = plt.figure(figsize=(20, 8))  

                # Plot für KMeans-Cluster
                ax1 = fig.add_subplot(121, projection='3d') 
                scatter1 = ax1.scatter(data_pca_3d[:, 0], data_pca_3d[:, 1], data_pca_3d[:, 2], c=df2['cluster'], cmap='viridis', s=50, alpha=0.6)
                ax1.set_title('3D-Visualisierung der KMeans-Cluster')
                ax1.set_xlabel('Erste Hauptkomponente')
                ax1.set_ylabel('Zweite Hauptkomponente')
                ax1.set_zlabel('Dritte Hauptkomponente')

                # Plot für tatsächliche Geste
                ax2 = fig.add_subplot(122, projection='3d')  
                scatter2 = ax2.scatter(data_pca_3d[:, 0], data_pca_3d[:, 1], data_pca_3d[:, 2], c=df2['class'], cmap='Set2', s=50, alpha=0.6)
                ax2.set_title('3D-Visualisierung der tatsächlichen Handgesten')
                ax2.set_xlabel('Erste Hauptkomponente')
                ax2.set_ylabel('Zweite Hauptkomponente')
                ax2.set_zlabel('Dritte Hauptkomponente')

                plt.tight_layout()
                plt.show()





                # t-Distributed Stochastic Neighbor Embedding (t-SNE) Visualisierung

                tsne = TSNE(n_components=2, random_state=0)
                data_tsne = tsne.fit_transform(data_array2)  # Alle Spalten außer 'cluster'

                plt.figure(figsize=(10, 6))

                plt.subplot(1, 2, 1)
                sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=df2['cluster'], palette='viridis')
                plt.title('t-SNE-Visualisierung der Cluster')
                plt.xlabel('t-SNE Feature 1')
                plt.ylabel('t-SNE Feature 2')

                plt.subplot(1, 2, 2)
                sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=df2['class'], palette='Set2')
                plt.title('t-SNE-Visualisierung der tatsächlichen Handgesten')
                plt.xlabel('t-SNE Feature 1')
                plt.ylabel('t-SNE Feature 2')
                plt.show()
kmeans()

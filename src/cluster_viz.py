import numpy as np
import umap.umap_ as umap
import hdbscan
from sklearn.ensemble import IsolationForest
import plotly.express as px
import pandas as pd

class DocumentClusterer:
    def __init__(self):
        """Inicializa os algoritmos de redução dimensional, clusterização e detecção de anomalias"""
        # Redução dimensional (384D -> 3D)
        self.reducer = umap.UMAP(
            n_components=3,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        
        # Clusterização hierárquica
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Detecção de anomalias
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42
        )

    def process_documents(self, embeddings: np.ndarray):
        """
        Processamento completo dos documentos:
        1. Redução dimensional
        2. Clusterização
        3. Detecção de anomalias
        
        Args:
            embeddings: Array numpy com os embeddings dos documentos (n_docs, 384)
            
        Returns:
            tuple: (clusters, reduced_embeddings, anomalies)
        """
        # Validação dos embeddings
        if embeddings.size == 0:
            raise ValueError("Array de embeddings vazio")
            
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # 1. Redução dimensional
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        # 2. Clusterização
        clusters = self.clusterer.fit_predict(reduced_embeddings)
        
        # 3. Detecção de anomalias
        anomalies = self.detect_anomalies(embeddings)
        
        return clusters, reduced_embeddings, anomalies

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduz a dimensionalidade dos embeddings para visualização 3D
        
        Args:
            embeddings: Array numpy com os embeddings originais
            
        Returns:
            np.ndarray: Embeddings reduzidos (n_docs, 3)
        """
        return self.reducer.fit_transform(embeddings)

    def detect_anomalies(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Identifica documentos anômalos usando Isolation Forest
        
        Args:
            embeddings: Array numpy com os embeddings originais
            
        Returns:
            np.ndarray: Array de anomalias (-1 para anomalias, 1 para normais)
        """
        return self.anomaly_detector.fit_predict(embeddings)

    def visualize_clusters(self, reduced_embeddings: np.ndarray, 
                         clusters: np.ndarray, 
                         documents: list,
                         anomalies: np.ndarray = None) -> px.scatter_3d:
        """
        Cria visualização 3D interativa dos clusters
        
        Args:
            reduced_embeddings: Embeddings reduzidos (n_docs, 3)
            clusters: Array com os labels de cluster
            documents: Lista com os textos originais
            anomalies: Array com as anomalias detectadas (opcional)
            
        Returns:
            plotly.graph_objects.Figure: Figura 3D interativa
        """
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'cluster': clusters,
            'text': [doc[:200] + '...' for doc in documents],
            'anomaly': anomalies if anomalies is not None else [0]*len(documents)
        })
        
        # Marca anomalias
        df['color'] = df.apply(
            lambda row: 'Anomaly' if row['anomaly'] == -1 else f'Cluster {row["cluster"]}',
            axis=1
        )
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='color',
            hover_data=['text'],
            title='Clusterização de Documentos Científicos',
            color_discrete_map={'Anomaly': 'red'},
            labels={'color': 'Categoria'}
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig

    # Método alternativo para compatibilidade
    def cluster_documents(self, embeddings: np.ndarray):
        """
        Versão simplificada para compatibilidade
        Retorna apenas clusters e embeddings reduzidos
        """
        clusters, reduced_embeddings, _ = self.process_documents(embeddings)
        return clusters, reduced_embeddings
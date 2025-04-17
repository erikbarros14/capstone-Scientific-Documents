import numpy as np
import umap.umap_ as umap
import hdbscan
from sklearn.ensemble import IsolationForest
import plotly.express as px
import pandas as pd
from typing import Tuple

class DocumentClusterer:
    def __init__(self):
        # Configurações padrão
        self.reducer = umap.UMAP(
            n_components=3,
            random_state=42,
            n_neighbors=3,  # Reduzido para trabalhar melhor com poucos docs
            min_dist=0.1,
            metric='cosine'
        )
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,  # Permite clusters menores
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42
        )

    def process_documents(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processamento adaptado para poucos documentos"""
        if len(embeddings) <= 2:  # Caso especial para 1-2 documentos
            clusters = np.zeros(len(embeddings), dtype=int)
            reduced_embeddings = np.random.rand(len(embeddings), 3) * 0.1  # Pequena variação
            anomalies = np.ones(len(embeddings))
            return clusters, reduced_embeddings, anomalies
        
        try:
            # Redução de dimensionalidade
            reduced_embeddings = self.reduce_dimensions(embeddings)
            
            # Clusterização
            clusters = self.clusterer.fit_predict(reduced_embeddings)
            
            # Detecção de anomalias
            anomalies = self.anomaly_detector.fit_predict(embeddings)
            
            return clusters, reduced_embeddings, anomalies
            
        except Exception as e:
            # Fallback para casos problemáticos
            print(f"Erro no processamento: {str(e)} - Usando fallback")
            clusters = np.zeros(len(embeddings), dtype=int)
            reduced_embeddings = np.random.rand(len(embeddings), 3)
            anomalies = np.ones(len(embeddings))
            return clusters, reduced_embeddings, anomalies

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Redução de dimensionalidade robusta"""
        if len(embeddings) <= 5:  # Ajuste especial para poucos docs
            self.reducer.n_neighbors = max(1, len(embeddings) - 1)
        return self.reducer.fit_transform(embeddings)

    def visualize_clusters(self, reduced_embeddings, clusters, documents, anomalies=None):
        """Visualização adaptativa"""
        if len(reduced_embeddings) == 1:
            return self._visualize_single_doc(reduced_embeddings[0], documents[0])
        
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'cluster': clusters,
            'text': [doc[:150] + '...' for doc in documents],
            'anomaly': anomalies if anomalies is not None else [1]*len(documents)
        })
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='cluster',
            symbol='anomaly',
            hover_data=['text'],
            title='Clusterização de Documentos'
        )
        return fig

    def _visualize_single_doc(self, coords, document):
        """Visualização para único documento"""
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=[coords[0]], y=[coords[1]], z=[coords[2]],
            mode='markers+text',
            marker=dict(size=10, color=['blue']),
            text=["Documento"],
            hovertext=[document[:200] + "..."]
        ))
        fig.update_layout(title="Visualização do Documento")
        return fig

    # Método alternativo para compatibilidade
    def cluster_documents(self, embeddings: np.ndarray):
        """
        Versão simplificada para compatibilidade
        Retorna apenas clusters e embeddings reduzidos
        """
        clusters, reduced_embeddings, _ = self.process_documents(embeddings)
        return clusters, reduced_embeddings
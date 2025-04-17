import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer

class SemanticSearcher:
    def __init__(self, collection, model_name='all-MiniLM-L6-v2'):
        self.collection = collection
        self.model = SentenceTransformer(model_name)  # Agora o modelo está dentro do searcher
    
    def search(self, query: str, top_k: int = 5):
        """Realiza busca semântica usando o modelo interno"""
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append({
                'content': doc,
                'source': meta['source'],
                'similarity': 1.0  # Placeholder, você pode calcular a similaridade real se quiser
            })
        
        return formatted_results
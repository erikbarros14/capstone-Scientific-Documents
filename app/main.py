import sys
import os
from pathlib import Path
import asyncio
import torch
torch.set_num_threads(1)

# Configuração segura do event loop para Streamlit
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

sys.path.append(str(Path(__file__).parent.parent))
import shutil
import streamlit as st
from src.ingest import DocumentIngestor
from src.cluster_viz import DocumentClusterer
from src.search import SemanticSearcher
import numpy as np
import pandas as pd
import chromadb
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import nltk

# Configuração da página
st.set_page_config(
    page_title="Explorador de Documentos Científicos", 
    layout="wide",
    page_icon="📚"
)

# Baixa stopwords uma vez no início
nltk.download('stopwords')
from nltk.corpus import stopwords

# Inicialização dos componentes
@st.cache_resource
def load_components():
    ingestor = DocumentIngestor()
    clusterer = DocumentClusterer()
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    try:
        collection = chroma_client.get_collection(name="scientific_papers")
    except:
        collection = chroma_client.create_collection(
            name="scientific_papers",
            metadata={"hnsw:space": "cosine"}
        )
    searcher = SemanticSearcher(collection)
    return ingestor, clusterer, collection, searcher

ingestor, clusterer, collection, searcher = load_components()

def create_single_document_viz(text, filename):
    """Cria visualização para um único documento"""
    fig = go.Figure(data=[go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=12, color=['blue']),
        text=[filename],
        hovertext=["Clique para ver detalhes"],
        textposition="middle center"
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        title=f"Visualização do Documento: {filename}",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def generate_insights(documents, clusters):
    """Gera insights automáticos sobre os documentos"""
    st.subheader("🔍 Insights Descobertos")
    
    with st.container():
        # Análise de clusters
        col1, col2 = st.columns(2)
        
        with col1:
            unique_clusters = len(set(clusters))
            if unique_clusters > 1:
                st.success(f"**{unique_clusters} Grupos Temáticos** identificados")
            else:
                st.info("Os documentos formam **um grupo coeso**")
        
        with col2:
            anomaly_count = sum(1 for c in clusters if c == -1)
            if anomaly_count > 0:
                st.warning(f"**{anomaly_count} Anomalias** detectadas")
            else:
                st.success("Nenhuma anomalia detectada")
        
        # Análise de conteúdo
        st.markdown("### 📊 Estatísticas de Texto")
        
        # Palavras mais frequentes
        words = [word for doc in documents 
                for word in doc.lower().split() 
                if word.isalpha() and word not in stopwords.words('portuguese')]
        
        if words:
            common_words = Counter(words).most_common(10)
            st.bar_chart(dict(common_words), 
                        use_container_width=True,
                        color="#4CAF50")
            
            # Tamanho médio
            avg_length = sum(len(doc) for doc in documents) / len(documents)
            st.metric("Tamanho Médio", f"{avg_length:.0f} caracteres")
        else:
            st.warning("Não foi possível analisar o conteúdo textual")

# Página principal
st.title("📚 Explorador de Documentos Científicos")

# Seção de upload na sidebar
with st.sidebar:
    st.header("Configurações")
    uploaded_files = st.file_uploader(
        "Carregue documentos científicos", 
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Suporta PDFs com texto, arquivos TXT e CSV"
    )

# Processamento principal
if uploaded_files:
    with st.spinner("Processando documentos..."):
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Salva os arquivos
        for file in uploaded_files:
            with open(os.path.join(temp_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        
        # Processa os documentos
        documents = []
        metadatas = []
        ids = []
        
        for idx, filename in enumerate(os.listdir(temp_dir)):
            if filename.lower().endswith(('.pdf', '.txt', '.csv')):
                filepath = os.path.join(temp_dir, filename)
                try:
                    processed = ingestor.process_file(filepath)
                    texts = [processed] if isinstance(processed, str) else processed
                    
                    for text in texts:
                        embedding = ingestor.generate_embedding(text)
                        doc_id = f"doc_{idx}_{len(documents)}"
                        
                        collection.add(
                            embeddings=[embedding],
                            documents=[text],
                            metadatas=[{"source": filename}],
                            ids=[doc_id]
                        )
                        
                        documents.append(text)
                        metadatas.append({
                            "source": filename,
                            "type": os.path.splitext(filename)[1][1:].upper(),
                            "length": len(text)
                        })
                        ids.append(doc_id)
                
                except Exception as e:
                    st.error(f"Erro ao processar {filename}: {str(e)}")

        if not documents:
            st.error("Nenhum documento válido encontrado.")
            shutil.rmtree(temp_dir)
            st.stop()

        # Obtém embeddings
        results = collection.get(ids=ids, include=["embeddings"])
        embeddings = np.array(results["embeddings"])
        
        # Visualização adaptada para 1 documento
        if len(documents) == 1:
            st.subheader("Visualização do Documento")
            fig = create_single_document_viz(documents[0], metadatas[0]["source"])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Conteúdo Completo")
            with st.expander(f"📄 {metadatas[0]['source']} - {metadatas[0]['length']} caracteres"):
                st.text(documents[0])
            
            st.subheader("Busca no Documento")
            query = st.text_input("Pesquise conceitos no documento:")
            if query:
                results = searcher.search(query)
                for res in results:
                    st.write(f"**Trecho relevante:**")
                    st.text(res['content'][:500] + "...")
            
            st.info("🔍 Para insights mais profundos, adicione mais documentos")
        
        else:
            # Processamento para múltiplos documentos
            clusters, reduced_embeddings, anomalies = clusterer.process_documents(embeddings)
            
            # Geração de insights
            generate_insights(documents, clusters)
            
            # Visualização 3D
            st.subheader("Visualização Interativa dos Clusters")
            fig = clusterer.visualize_clusters(reduced_embeddings, clusters, documents, anomalies)
            st.plotly_chart(fig, use_container_width=True)
            
            # Lista de Documentos
            st.subheader("Documentos Processados")
            for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
                with st.expander(f"{idx+1}. {meta['source']} ({meta['type']}, {meta['length']} chars)"):
                    st.text(doc[:2000] + ("..." if len(doc) > 2000 else ""))
        
        # Estatísticas na Sidebar
        with st.sidebar:
            st.header("Estatísticas")
            st.metric("Documentos processados", len(documents))
            st.metric("Tipo principal", metadatas[0]["type"])
            
            if len(documents) > 1 and 'clusters' in locals():
                st.metric("Clusters identificados", len(set(clusters)))
            
            # Limpeza
            shutil.rmtree(temp_dir)

else:
    # Página inicial
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        ## Bem-vindo ao Explorador de Documentos Científicos
            
        **Como usar:**
        1. Carregue documentos científicos
        2. Explore relações entre conteúdos
        3. Descubra insights automáticos
        """)
        st.info("Experimente carregar PDFs, TXT ou CSV com conteúdo textual")
    with col2:
        st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80",
               caption="Análise inteligente de documentos")

st.markdown("---")
st.caption("🔍 Sistema de análise de documentos científicos | v1.0")
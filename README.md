📌 Visão Geral
O Scientific Documents Explorer é uma ferramenta avançada de análise de documentos científicos que utiliza Processamento de Linguagem Natural (NLP) e Machine Learning para organizar, visualizar e explorar coleções de artigos acadêmicos de forma inteligente.

Screenshot da Interface

✨ Principais Recursos
Processamento Automático de documentos em PDF, TXT e CSV

Clusterização Inteligente de documentos por similaridade semântica

Visualização 3D Interativa dos clusters de documentos

Busca Semântica que vai além de palavras-chave

Detecção de Anomalias para identificar documentos atípicos

Geração Automática de Insights sobre a coleção de documentos

🛠️ Tecnologias Utilizadas
Tecnologia	Finalidade
Python 3.10+	Linguagem principal
Streamlit	Interface web
ChromaDB	Banco de dados vetorial
Sentence Transformers	Geração de embeddings
UMAP	Redução de dimensionalidade
HDBSCAN	Clusterização
Plotly	Visualização 3D
NLTK	Processamento de texto
🚀 Como Executar o Projeto
Pré-requisitos
Python 3.10 ou superior

Pip instalado

Instalação
Clone o repositório:

bash
git clone https://github.com/erikbarros14/capstone-Scientific-Documents/tree/main
cd scientific-documents-explorer
Crie e ative um ambiente virtual (recomendado):

bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
Instale as dependências:

bash
pip install -r requirements.txt
Baixe os modelos necessários:

bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
Execução
bash
streamlit run main.py
O aplicativo estará disponível em http://localhost:8501

🔍 Funcionamento Detalhado
1. Ingestão de Documentos
O sistema processa três tipos de arquivos:

PDF: Extrai texto com PyPDF2

TXT: Lê conteúdo diretamente

CSV: Trata cada linha como documento separado


# Trecho do ingest.py (linhas 58-99)
def process_file(self, filepath):
    if filepath.endswith('.pdf'):
        text = self._extract_from_pdf(filepath)
    elif filepath.endswith('.txt'):
        text = self._read_text_file(filepath)
    elif filepath.endswith('.csv'):
        text = self._process_csv(filepath)
    return self._clean_text(text)
2. Geração de Embeddings
Utiliza o modelo all-MiniLM-L6-v2 para criar representações vetoriais de 384 dimensões:

# Trecho do ingest.py (linhas 8-44)
def generate_embedding(self, text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)
3. Clusterização
Combina UMAP para redução dimensional e HDBSCAN para clusterização:

# Trecho do cluster_viz.py (linhas 10-24)
def process_documents(self, embeddings):
    umap_embeddings = umap.UMAP(n_components=3).fit_transform(embeddings)
    clusters = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(umap_embeddings)
    anomalies = self._detect_anomalies(umap_embeddings)
    return clusters, umap_embeddings, anomalies
4. Visualização Interativa
Interface construída com Streamlit e Plotly:


# Trecho do main.py (linhas 148-178)
def visualize_clusters(self, embeddings, clusters, documents, anomalies):
    fig = px.scatter_3d(
        x=embeddings[:,0], y=embeddings[:,1], z=embeddings[:,2],
        color=clusters, hover_text=documents
    )
    fig.update_traces(marker=dict(size=5))
    return fig
📊 Demonstração
Upload de Documentos:

Selecione múltiplos arquivos PDF, TXT ou CSV

O sistema processará automaticamente

Exploração Visual:

Documentos similares aparecem agrupados no espaço 3D

Clusters são coloridos automaticamente

Anomalias aparecem destacadas

Busca Semântica:

Digite um conceito (ex: "machine learning")

O sistema retorna documentos relacionados semanticamente

Filtros:

Filtre por tipo de arquivo, cluster ou tamanho do texto

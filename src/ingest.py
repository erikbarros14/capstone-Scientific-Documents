import os
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Union
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Configuração segura do event loop
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            self.model = SentenceTransformer(model_name)
            test_embed = self.model.encode("test")
            if len(test_embed) == 0:
                raise RuntimeError("Modelo não gerou embedding de teste")
                
            self.chroma_client = chromadb.PersistentClient(path="chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="scientific_papers",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("DocumentIngestor inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Falha na inicialização: {str(e)}")
            raise RuntimeError(f"Erro crítico: {str(e)}")

    # ... (mantenha o resto dos métodos igual)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto com verificação de PDF válido"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if len(reader.pages) == 0:
                    raise ValueError("PDF sem páginas legíveis")
                text = " ".join([page.extract_text() or "" for page in reader.pages])
                if not text.strip():
                    raise ValueError("PDF não contém texto extraível")
                return text
        except Exception as e:
            logger.error(f"Erro ao extrair PDF {pdf_path}: {str(e)}")
            raise

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extrai texto de arquivo TXT"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if not text.strip():
                    raise ValueError("Arquivo TXT vazio")
                return text
        except Exception as e:
            logger.error(f"Erro ao ler TXT {txt_path}: {str(e)}")
            raise

    def extract_text_from_csv(self, csv_path: str) -> List[str]:
        """Extrai texto de CSV"""
        try:
            df = pd.read_csv(csv_path)
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) == 0:
                text_columns = df.columns
            
            return [" ".join(str(row[col]) for col in text_columns) for _, row in df.iterrows()]
        except Exception as e:
            logger.error(f"Erro ao processar CSV {csv_path}: {str(e)}")
            raise

    def process_file(self, file_path: str) -> Union[str, List[str]]:
        """Processa arquivo de acordo com a extensão"""
        try:
            if file_path.lower().endswith('.pdf'):
                return self.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                return self.extract_text_from_txt(file_path)
            elif file_path.lower().endswith('.csv'):
                return self.extract_text_from_csv(file_path)
            else:
                raise ValueError(f"Formato não suportado: {file_path}")
        except Exception as e:
            logger.error(f"Falha ao processar {file_path}: {str(e)}")
            raise

    def generate_embedding(self, text: str):
        """Gera embedding com verificação"""
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            if len(embedding) == 0:
                raise ValueError("Embedding vazio")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Falha ao gerar embedding: {str(e)}")
            raise

    def ingest_documents(self, folder_path: str):
        """Processa documentos com tratamento de erros"""
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")
                
            documents, metadatas, ids = [], [], []
            
            for idx, filename in enumerate(os.listdir(folder_path)):
                if filename.lower().endswith(('.pdf', '.txt', '.csv')):
                    filepath = os.path.join(folder_path, filename)
                    try:
                        text = self.process_file(filepath)
                        texts = [text] if isinstance(text, str) else text
                        
                        for text_content in texts:
                            try:
                                embedding = self.generate_embedding(text_content)
                                doc_id = f"doc_{idx}_{len(documents)}"
                                
                                self.collection.add(
                                    embeddings=[embedding],
                                    documents=[text_content],
                                    metadatas=[{"source": filename}],
                                    ids=[doc_id]
                                )
                                
                                documents.append(text_content)
                                metadatas.append({"source": filename})
                                ids.append(doc_id)
                                
                            except Exception as e:
                                logger.warning(f"Documento {filename} ignorado: {str(e)}")
                                
                    except Exception as e:
                        logger.error(f"Falha no arquivo {filename}: {str(e)}")
            
            if not documents:
                raise RuntimeError("Nenhum documento válido foi processado")
                
            return documents, metadatas, ids
            
        except Exception as e:
            logger.error(f"Erro fatal: {str(e)}")
            raise
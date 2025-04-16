import streamlit as st
from typing import List, Dict

def search_interface(searcher) -> List[Dict]:
    """Componente de interface para busca semântica"""
    st.sidebar.header("Busca Semântica")
    
    search_query = st.sidebar.text_input(
        "Digite sua busca:",
        placeholder="Ex: machine learning applications in healthcare"
    )
    
    top_k = st.sidebar.slider(
        "Número de resultados:",
        min_value=1,
        max_value=20,
        value=5
    )
    
    if search_query:
        with st.spinner("Buscando documentos similares..."):
            results = searcher.search(search_query, top_k=top_k)
            
            st.subheader(f"Resultados para: '{search_query}'")
            for i, result in enumerate(results, 1):
                with st.expander(f"Resultado {i}: {result['source']}"):
                    st.write(result['content'][:1000] + "...")
            
            return results
    
    return []
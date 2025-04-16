import streamlit as st
from typing import Dict, Any

def create_filters(cluster_labels: list, metadata: list) -> Dict[str, Any]:
    """Cria filtros interativos para a visualização"""
    st.sidebar.header("Filtros")
    
    filters = {}
    
    # Filtro por cluster
    unique_clusters = sorted(set(cluster_labels))
    selected_clusters = st.sidebar.multiselect(
        "Selecione clusters",
        options=unique_clusters,
        default=unique_clusters
    )
    filters['clusters'] = selected_clusters
    
    # Filtro por fonte
    if metadata and 'source' in metadata[0]:
        sources = sorted(set(m['source'] for m in metadata))
        selected_sources = st.sidebar.multiselect(
            "Filtrar por fonte",
            options=sources,
            default=sources
        )
        filters['sources'] = selected_sources
    
    # Filtro por tipo de documento
    if metadata and 'type' in metadata[0]:
        doc_types = sorted(set(m['type'] for m in metadata))
        selected_types = st.sidebar.multiselect(
            "Filtrar por tipo de documento",
            options=doc_types,
            default=doc_types
        )
        filters['types'] = selected_types
    
    # Filtro por tamanho do documento
    if metadata and 'length' in metadata[0]:
        max_length = max(m['length'] for m in metadata)
        doc_lengths = st.sidebar.slider(
            "Filtrar por tamanho (caracteres)",
            min_value=0,
            max_value=max_length,
            value=(0, max_length)
        )
        filters['length_range'] = doc_lengths
    
    return filters


def apply_filters(data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica os filtros selecionados aos dados"""
    filtered_data = data.copy()
    
    # Aplica filtros sequencialmente
    filter_order = ['clusters', 'sources', 'types', 'length_range']
    
    for filter_key in filter_order:
        if filter_key not in filters:
            continue
            
        if filter_key == 'clusters':
            mask = [label in filters['clusters'] for label in data['clusters']]
        elif filter_key == 'sources':
            mask = [source in filters['sources'] for source in data['sources']]
        elif filter_key == 'types':
            mask = [doc_type in filters['types'] for doc_type in data['types']]
        elif filter_key == 'length_range':
            min_len, max_len = filters['length_range']
            mask = [min_len <= length <= max_len for length in data['lengths']]
        else:
            continue
            
        # Aplica a máscara a todos os dados
        for key in filtered_data:
            if isinstance(filtered_data[key], list) and len(filtered_data[key]) == len(mask):
                filtered_data[key] = [val for val, m in zip(filtered_data[key], mask) if m]
    
    return filtered_data
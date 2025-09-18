"""
Módulo para algoritmos de clustering K-means e K-medoids com análise do índice de Dunn.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from typing import Tuple, Dict, List, Any
from config import OUTPUTS_DIR, RANDOM_STATE


class KMedoids:
    """
    Implementação do algoritmo K-medoids usando PAM (Partitioning Around Medoids).
    """
    
    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: int = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit(self, X: np.ndarray) -> 'KMedoids':
        """Ajusta o modelo K-medoids aos dados."""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Inicialização aleatória dos medoides
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # Calcular matriz de distâncias
        distances = pairwise_distances(X, metric='euclidean')
        
        for iteration in range(self.max_iter):
            # Atribuir pontos aos clusters mais próximos
            labels = np.argmin(distances[medoid_indices], axis=0)
            
            # Atualizar medoides
            new_medoid_indices = []
            for k in range(self.n_clusters):
                cluster_mask = (labels == k)
                if np.sum(cluster_mask) == 0:
                    # Se cluster vazio, manter medoide atual
                    new_medoid_indices.append(medoid_indices[k])
                    continue
                    
                cluster_points = np.where(cluster_mask)[0]
                # Encontrar ponto que minimiza distância total dentro do cluster
                min_cost = float('inf')
                best_medoid = medoid_indices[k]
                
                for candidate in cluster_points:
                    cost = np.sum(distances[candidate][cluster_points])
                    if cost < min_cost:
                        min_cost = cost
                        best_medoid = candidate
                        
                new_medoid_indices.append(best_medoid)
            
            new_medoid_indices = np.array(new_medoid_indices)
            
            # Verificar convergência
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
                
            medoid_indices = new_medoid_indices
        
        self.cluster_centers_ = X[medoid_indices]
        self.labels_ = labels
        
        # Calcular inertia (soma das distâncias aos medoides)
        self.inertia_ = 0
        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            if np.sum(cluster_mask) > 0:
                cluster_distances = distances[medoid_indices[k]][cluster_mask]
                self.inertia_ += np.sum(cluster_distances)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz os clusters para novos dados."""
        distances_to_medoids = pairwise_distances(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(distances_to_medoids, axis=1)


def calculate_dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula o índice de Dunn para avaliar a qualidade do clustering.
    
    Índice de Dunn = min(distância entre clusters) / max(distância intra-cluster)
    Valores maiores indicam melhor clustering.
    """
    n_clusters = len(np.unique(labels))
    
    if n_clusters <= 1:
        return 0.0
    
    # Calcular distâncias entre clusters (mínima distância entre pontos de clusters diferentes)
    min_inter_cluster_distance = float('inf')
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i_points = X[labels == i]
            cluster_j_points = X[labels == j]
            
            if len(cluster_i_points) == 0 or len(cluster_j_points) == 0:
                continue
                
            # Distância mínima entre os clusters
            distances = pairwise_distances(cluster_i_points, cluster_j_points, metric='euclidean')
            min_distance = np.min(distances)
            min_inter_cluster_distance = min(min_inter_cluster_distance, min_distance)
    
    # Calcular máxima distância intra-cluster
    max_intra_cluster_distance = 0.0
    
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) <= 1:
            continue
            
        # Distância máxima dentro do cluster
        distances = pairwise_distances(cluster_points, metric='euclidean')
        max_distance = np.max(distances)
        max_intra_cluster_distance = max(max_intra_cluster_distance, max_distance)
    
    if max_intra_cluster_distance == 0:
        return 0.0
    
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance
    return dunn_index


def apply_clustering_analysis(
    data_path: str,
    data_name: str,
    k_values: List[int],
    algorithms: List[str] = ['kmeans', 'kmedoids']
) -> Dict[str, Any]:
    """
    Aplica algoritmos de clustering para diferentes valores de K e calcula métricas.
    
    Args:
        data_path: Caminho para o arquivo CSV com os dados
        data_name: Nome descritivo dos dados
        k_values: Lista de valores K para testar
        algorithms: Lista de algoritmos ('kmeans', 'kmedoids')
    
    Returns:
        Dicionário com resultados da análise
    """
    print(f"\n=== Análise de Clustering: {data_name} ===")
    
    # Carregar dados
    df = pd.read_csv(data_path)
    
    # Filtrar apenas colunas numéricas (excluir metadados como image_path, folder, filename)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remover sample_id se existir (é apenas identificador)
    if 'sample_id' in numeric_columns:
        numeric_columns.remove('sample_id')
    
    X = df[numeric_columns].values
    n_samples, n_features = X.shape
    
    print(f"Colunas numéricas utilizadas: {numeric_columns}")
    
    print(f"Dados carregados: {n_samples} amostras, {n_features} dimensões")
    
    results = {
        'data_name': data_name,
        'data_shape': X.shape,
        'k_values': k_values,
        'algorithms': algorithms,
        'metrics': {}
    }
    
    for algorithm in algorithms:
        print(f"\n--- Aplicando {algorithm.upper()} ---")
        results['metrics'][algorithm] = {
            'dunn_index': [],
            'inertia': [],
            'labels': {},
            'centers': {}
        }
        
        for k in k_values:
            print(f"K = {k}...", end=' ')
            
            try:
                if algorithm == 'kmeans':
                    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                elif algorithm == 'kmedoids':
                    model = KMedoids(n_clusters=k, random_state=RANDOM_STATE)
                else:
                    raise ValueError(f"Algoritmo desconhecido: {algorithm}")
                
                # Ajustar modelo
                model.fit(X)
                labels = model.labels_
                
                # Calcular métricas
                dunn_idx = calculate_dunn_index(X, labels)
                inertia = model.inertia_
                
                # Armazenar resultados
                results['metrics'][algorithm]['dunn_index'].append(dunn_idx)
                results['metrics'][algorithm]['inertia'].append(inertia)
                results['metrics'][algorithm]['labels'][str(k)] = labels.tolist()
                results['metrics'][algorithm]['centers'][str(k)] = model.cluster_centers_.tolist()
                
                print(f"Dunn: {dunn_idx:.4f}, Inertia: {inertia:.2f}")
                
            except Exception as e:
                print(f"Erro: {e}")
                results['metrics'][algorithm]['dunn_index'].append(0.0)
                results['metrics'][algorithm]['inertia'].append(float('inf'))
                results['metrics'][algorithm]['labels'][str(k)] = []
                results['metrics'][algorithm]['centers'][str(k)] = []
    
    return results


def create_clustering_visualizations(
    data_path: str,
    results: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Cria visualizações dos resultados de clustering.
    """
    df = pd.read_csv(data_path)
    
    # Filtrar apenas colunas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'sample_id' in numeric_columns:
        numeric_columns.remove('sample_id')
    
    X = df[numeric_columns].values
    data_name = results['data_name']
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gráfico do índice de Dunn
    fig_dunn = go.Figure()
    
    for algorithm in results['algorithms']:
        fig_dunn.add_trace(go.Scatter(
            x=results['k_values'],
            y=results['metrics'][algorithm]['dunn_index'],
            mode='lines+markers',
            name=f'{algorithm.upper()}',
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig_dunn.update_layout(
        title=f'Índice de Dunn - {data_name}',
        xaxis_title='Número de Clusters (K)',
        yaxis_title='Índice de Dunn',
        template='plotly_white',
        height=500,
        font=dict(size=12)
    )
    
    dunn_path = os.path.join(output_dir, f'dunn_index_{data_name.lower().replace(" ", "_")}.html')
    fig_dunn.write_html(dunn_path)
    
    # 2. Gráfico da Inertia
    fig_inertia = go.Figure()
    
    for algorithm in results['algorithms']:
        fig_inertia.add_trace(go.Scatter(
            x=results['k_values'],
            y=results['metrics'][algorithm]['inertia'],
            mode='lines+markers',
            name=f'{algorithm.upper()}',
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig_inertia.update_layout(
        title=f'Inertia (WCSS) - {data_name}',
        xaxis_title='Número de Clusters (K)',
        yaxis_title='Inertia',
        template='plotly_white',
        height=500,
        font=dict(size=12)
    )
    
    inertia_path = os.path.join(output_dir, f'inertia_{data_name.lower().replace(" ", "_")}.html')
    fig_inertia.write_html(inertia_path)
    
    # 3. Visualização dos clusters para valores selecionados de K
    n_features = X.shape[1]
    
    # Encontrar melhor K baseado no índice de Dunn
    best_k_per_algorithm = {}
    for algorithm in results['algorithms']:
        dunn_values = results['metrics'][algorithm]['dunn_index']
        if max(dunn_values) > 0:
            best_idx = np.argmax(dunn_values)
            best_k_per_algorithm[algorithm] = results['k_values'][best_idx]
        else:
            best_k_per_algorithm[algorithm] = results['k_values'][len(results['k_values'])//2]
    
    # Visualizar clusters
    visualize_clusters_for_best_k(X, results, best_k_per_algorithm, output_dir, data_name)
    
    print(f"Visualizações salvas em: {output_dir}")


def visualize_clusters_for_best_k(
    X: np.ndarray,
    results: Dict[str, Any],
    best_k_per_algorithm: Dict[str, int],
    output_dir: str,
    data_name: str
) -> None:
    """
    Visualiza os clusters para os melhores valores de K encontrados.
    """
    n_features = X.shape[1]
    
    for algorithm, best_k in best_k_per_algorithm.items():
        labels = np.array(results['metrics'][algorithm]['labels'][str(best_k)])
        
        if len(labels) == 0:
            continue
            
        print(f"Visualizando {algorithm} com K={best_k}")
        
        if n_features == 2:
            # Dados 2D - visualização direta
            fig = px.scatter(
                x=X[:, 0], y=X[:, 1],
                color=labels.astype(str),
                title=f'Clusters {algorithm.upper()} (K={best_k}) - {data_name}',
                labels={'x': 'Dimensão 1', 'y': 'Dimensão 2'},
                template='plotly_white'
            )
            
        elif n_features == 3:
            # Dados 3D - visualização direta
            fig = px.scatter_3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                color=labels.astype(str),
                title=f'Clusters {algorithm.upper()} (K={best_k}) - {data_name}',
                labels={'x': 'Dimensão 1', 'y': 'Dimensão 2', 'z': 'Dimensão 3'},
                template='plotly_white'
            )
            
        else:
            # Dados >3D - reduzir para 2D com PCA
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(X)
            
            explained_var = pca.explained_variance_ratio_
            
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=labels.astype(str),
                title=f'Clusters {algorithm.upper()} (K={best_k}) - {data_name}<br>'
                      f'PCA: {explained_var[0]:.1%} + {explained_var[1]:.1%} = {sum(explained_var):.1%} da variância',
                labels={'x': f'PC1 ({explained_var[0]:.1%})', 'y': f'PC2 ({explained_var[1]:.1%})'},
                template='plotly_white'
            )
        
        # Salvar visualização
        cluster_path = os.path.join(
            output_dir, 
            f'clusters_{algorithm}_{data_name.lower().replace(" ", "_")}_k{best_k}.html'
        )
        fig.write_html(cluster_path)


def save_clustering_results(results: Dict[str, Any], output_path: str) -> None:
    """Salva os resultados de clustering em arquivo JSON."""
    # Converter arrays numpy para listas para serialização JSON
    results_copy = results.copy()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados salvos em: {output_path}")


def print_clustering_summary(results: Dict[str, Any]) -> None:
    """Imprime um resumo dos resultados de clustering."""
    data_name = results['data_name']
    print(f"\n{'='*60}")
    print(f"RESUMO DOS RESULTADOS - {data_name}")
    print(f"{'='*60}")
    
    for algorithm in results['algorithms']:
        print(f"\n--- {algorithm.upper()} ---")
        dunn_values = results['metrics'][algorithm]['dunn_index']
        inertia_values = results['metrics'][algorithm]['inertia']
        
        if max(dunn_values) > 0:
            best_idx = np.argmax(dunn_values)
            best_k = results['k_values'][best_idx]
            best_dunn = dunn_values[best_idx]
            best_inertia = inertia_values[best_idx]
            
            print(f"Melhor K: {best_k}")
            print(f"Melhor Índice de Dunn: {best_dunn:.4f}")
            print(f"Inertia correspondente: {best_inertia:.2f}")
        else:
            print("Nenhum resultado válido encontrado")
        
        print(f"Todos os índices de Dunn: {[f'{x:.4f}' for x in dunn_values]}")
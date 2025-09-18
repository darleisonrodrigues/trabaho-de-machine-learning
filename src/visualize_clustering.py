"""
Módulo para visualização avançada dos resultados de clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os
from typing import Dict, List, Any, Tuple
from config import OUTPUTS_DIR, RANDOM_STATE


def create_comprehensive_clustering_dashboard(
    results_list: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Cria um dashboard abrangente comparando resultados de clustering entre diferentes embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Comparação dos índices de Dunn entre embeddings
    create_dunn_comparison_plot(results_list, output_dir)
    
    # 2. Heatmap dos resultados
    create_results_heatmap(results_list, output_dir)
    
    # 3. Análise do cotovelo (Elbow method)
    create_elbow_analysis(results_list, output_dir)
    
    # 4. Dashboard interativo principal
    create_main_dashboard(results_list, output_dir)


def create_dunn_comparison_plot(results_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Cria gráfico comparativo dos índices de Dunn entre diferentes embeddings."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['K-means', 'K-medoids', 'Comparação K-means', 'Comparação K-medoids'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    # Subplots para cada algoritmo
    for i, algorithm in enumerate(['kmeans', 'kmedoids']):
        row = 1
        col = i + 1
        
        for j, results in enumerate(results_list):
            if algorithm in results['algorithms']:
                dunn_values = results['metrics'][algorithm]['dunn_index']
                k_values = results['k_values']
                
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=dunn_values,
                        mode='lines+markers',
                        name=f"{results['data_name']} - {algorithm}",
                        line=dict(color=colors[j % len(colors)], width=2),
                        marker=dict(size=6),
                        showlegend=(i == 0)  # Mostrar legenda apenas no primeiro subplot
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Número de Clusters (K)", row=row, col=col)
        fig.update_yaxes(title_text="Índice de Dunn", row=row, col=col)
    
    # Comparações diretas
    for i, algorithm in enumerate(['kmeans', 'kmedoids']):
        row = 2
        col = i + 1
        
        # Encontrar melhor embedding para cada K
        all_k_values = results_list[0]['k_values']  # Assumindo que todos têm os mesmos K
        
        best_dunn_per_k = []
        best_embedding_per_k = []
        
        for k_idx, k in enumerate(all_k_values):
            best_dunn = -1
            best_embedding = ""
            
            for results in results_list:
                if algorithm in results['algorithms']:
                    dunn_value = results['metrics'][algorithm]['dunn_index'][k_idx]
                    if dunn_value > best_dunn:
                        best_dunn = dunn_value
                        best_embedding = results['data_name']
            
            best_dunn_per_k.append(best_dunn)
            best_embedding_per_k.append(best_embedding)
        
        # Plotar linha do melhor resultado
        fig.add_trace(
            go.Scatter(
                x=all_k_values,
                y=best_dunn_per_k,
                mode='lines+markers',
                name=f"Melhor {algorithm}",
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, color='red'),
                text=best_embedding_per_k,
                textposition="top center",
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Número de Clusters (K)", row=row, col=col)
        fig.update_yaxes(title_text="Melhor Índice de Dunn", row=row, col=col)
    
    fig.update_layout(
        title="Comparação de Índices de Dunn entre Embeddings",
        height=800,
        template='plotly_white',
        font=dict(size=10)
    )
    
    comparison_path = os.path.join(output_dir, 'dunn_comparison_dashboard.html')
    fig.write_html(comparison_path)
    print(f"Dashboard de comparação salvo em: {comparison_path}")


def create_results_heatmap(results_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Cria heatmap dos resultados de clustering."""
    
    # Preparar dados para o heatmap
    algorithms = ['kmeans', 'kmedoids']
    k_values = results_list[0]['k_values']
    embeddings = [r['data_name'] for r in results_list]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['K-means - Índice de Dunn', 'K-medoids - Índice de Dunn'],
        horizontal_spacing=0.1
    )
    
    for i, algorithm in enumerate(algorithms):
        # Criar matriz de resultados
        heatmap_data = []
        
        for results in results_list:
            if algorithm in results['algorithms']:
                dunn_values = results['metrics'][algorithm]['dunn_index']
                heatmap_data.append(dunn_values)
            else:
                heatmap_data.append([0] * len(k_values))
        
        heatmap_data = np.array(heatmap_data)
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=[f"K={k}" for k in k_values],
                y=embeddings,
                colorscale='Viridis',
                text=np.round(heatmap_data, 4),
                texttemplate="%{text}",
                textfont={"size": 8},
                name=algorithm,
                showscale=(i == 1)  # Mostrar escala apenas no último
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Heatmap dos Índices de Dunn",
        height=400,
        template='plotly_white'
    )
    
    heatmap_path = os.path.join(output_dir, 'clustering_heatmap.html')
    fig.write_html(heatmap_path)
    print(f"Heatmap salvo em: {heatmap_path}")


def create_elbow_analysis(results_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Cria análise do método do cotovelo (Elbow method) para determinar K ótimo."""
    
    fig = make_subplots(
        rows=len(results_list), cols=2,
        subplot_titles=[item for results in results_list 
                       for item in [f"{results['data_name']} - K-means", 
                                   f"{results['data_name']} - K-medoids"]],
        vertical_spacing=0.05
    )
    
    for i, results in enumerate(results_list):
        k_values = results['k_values']
        
        for j, algorithm in enumerate(['kmeans', 'kmedoids']):
            if algorithm in results['algorithms']:
                inertia_values = results['metrics'][algorithm]['inertia']
                
                # Calcular diferenças para identificar o "cotovelo"
                differences = np.diff(inertia_values)
                second_differences = np.diff(differences)
                
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=inertia_values,
                        mode='lines+markers',
                        name=f"{results['data_name']} {algorithm}",
                        line=dict(width=3),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=i+1, col=j+1
                )
                
                # Destacar possível cotovelo (maior segunda derivada)
                if len(second_differences) > 0:
                    elbow_idx = np.argmax(np.abs(second_differences)) + 2  # +2 devido aos diffs
                    if elbow_idx < len(k_values):
                        fig.add_trace(
                            go.Scatter(
                                x=[k_values[elbow_idx]],
                                y=[inertia_values[elbow_idx]],
                                mode='markers',
                                marker=dict(size=15, color='red', symbol='star'),
                                name=f"Cotovelo K={k_values[elbow_idx]}",
                                showlegend=False
                            ),
                            row=i+1, col=j+1
                        )
                
                fig.update_xaxes(title_text="K", row=i+1, col=j+1)
                fig.update_yaxes(title_text="Inertia", row=i+1, col=j+1)
    
    fig.update_layout(
        title="Análise do Método do Cotovelo",
        height=300 * len(results_list),
        template='plotly_white'
    )
    
    elbow_path = os.path.join(output_dir, 'elbow_analysis.html')
    fig.write_html(elbow_path)
    print(f"Análise do cotovelo salva em: {elbow_path}")


def create_main_dashboard(results_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Cria o dashboard principal interativo com todos os resultados."""
    
    # Preparar dados para tabela de resumo
    summary_data = []
    
    for results in results_list:
        for algorithm in results['algorithms']:
            dunn_values = results['metrics'][algorithm]['dunn_index']
            inertia_values = results['metrics'][algorithm]['inertia']
            k_values = results['k_values']
            
            if max(dunn_values) > 0:
                best_idx = np.argmax(dunn_values)
                best_k = k_values[best_idx]
                best_dunn = dunn_values[best_idx]
                best_inertia = inertia_values[best_idx]
                
                summary_data.append({
                    'Embedding': results['data_name'],
                    'Algoritmo': algorithm.upper(),
                    'Melhor K': best_k,
                    'Índice de Dunn': f"{best_dunn:.4f}",
                    'Inertia': f"{best_inertia:.2f}",
                    'Dimensões': results['data_shape'][1]
                })
    
    # Criar figura com subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Índices de Dunn - K-means', 'Índices de Dunn - K-medoids',
            'Inertia - K-means', 'Inertia - K-medoids',
            'Resumo dos Melhores Resultados', ''
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "table", "colspan": 2}, None]],
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    # Plotar resultados
    for i, algorithm in enumerate(['kmeans', 'kmedoids']):
        # Índices de Dunn
        for j, results in enumerate(results_list):
            if algorithm in results['algorithms']:
                dunn_values = results['metrics'][algorithm]['dunn_index']
                k_values = results['k_values']
                
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=dunn_values,
                        mode='lines+markers',
                        name=f"{results['data_name']}",
                        line=dict(color=colors[j % len(colors)], width=2),
                        marker=dict(size=6),
                        legendgroup=f"group{j}",
                        showlegend=(i == 0)
                    ),
                    row=1, col=i+1
                )
        
        # Inertia
        for j, results in enumerate(results_list):
            if algorithm in results['algorithms']:
                inertia_values = results['metrics'][algorithm]['inertia']
                k_values = results['k_values']
                
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=inertia_values,
                        mode='lines+markers',
                        name=f"{results['data_name']}",
                        line=dict(color=colors[j % len(colors)], width=2),
                        marker=dict(size=6),
                        legendgroup=f"group{j}",
                        showlegend=False
                    ),
                    row=2, col=i+1
                )
    
    # Tabela de resumo
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(df_summary.columns),
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[df_summary[col] for col in df_summary.columns],
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=3, col=1
        )
    
    # Atualizar layout
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Número de Clusters (K)", row=i, col=j)
            
    fig.update_yaxes(title_text="Índice de Dunn", row=1, col=1)
    fig.update_yaxes(title_text="Índice de Dunn", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=2, col=1)
    fig.update_yaxes(title_text="Inertia", row=2, col=2)
    
    fig.update_layout(
        title="Dashboard Principal - Análise de Clustering",
        height=1200,
        template='plotly_white',
        font=dict(size=10)
    )
    
    dashboard_path = os.path.join(output_dir, 'clustering_main_dashboard.html')
    fig.write_html(dashboard_path)
    print(f"Dashboard principal salvo em: {dashboard_path}")


def create_cluster_distribution_analysis(
    data_path: str,
    results: Dict[str, Any],
    output_dir: str
) -> None:
    """Cria análise da distribuição dos clusters."""
    
    df = pd.read_csv(data_path)
    
    # Filtrar apenas colunas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'sample_id' in numeric_columns:
        numeric_columns.remove('sample_id')
    
    X = df[numeric_columns].values
    data_name = results['data_name']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analisar distribuição para os melhores K de cada algoritmo
    # Criar figura com subplots (barras e pizza separados)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['K-means - Distribuição', 'K-medoids - Distribuição',
                       'K-means - Tamanhos (%)', 'K-medoids - Tamanhos (%)'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "pie"}]],
        vertical_spacing=0.15
    )
    
    for i, algorithm in enumerate(['kmeans', 'kmedoids']):
        if algorithm not in results['algorithms']:
            continue
            
        dunn_values = results['metrics'][algorithm]['dunn_index']
        if max(dunn_values) <= 0:
            continue
            
        best_idx = np.argmax(dunn_values)
        best_k = results['k_values'][best_idx]
        labels = np.array(results['metrics'][algorithm]['labels'][str(best_k)])
        
        # Distribuição dos clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {label}" for label in unique_labels],
                y=counts,
                name=f"{algorithm} (K={best_k})",
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Tamanhos dos clusters como percentual
        percentages = (counts / len(labels)) * 100
        
        fig.add_trace(
            go.Pie(
                labels=[f"Cluster {label}" for label in unique_labels],
                values=percentages,
                name=f"{algorithm} %",
                showlegend=False
            ),
            row=2, col=i+1
        )
    
    fig.update_layout(
        title=f"Análise de Distribuição dos Clusters - {data_name}",
        height=600,
        template='plotly_white'
    )
    
    distribution_path = os.path.join(output_dir, f'cluster_distribution_{data_name.lower().replace(" ", "_")}.html')
    fig.write_html(distribution_path)
    print(f"Análise de distribuição salva em: {distribution_path}")


def create_silhouette_analysis(data_path: str, results: Dict[str, Any], output_dir: str) -> None:
    """Cria análise de silhouette para complementar o índice de Dunn."""
    try:
        from sklearn.metrics import silhouette_score, silhouette_samples
        
        df = pd.read_csv(data_path)
        
        # Filtrar apenas colunas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'sample_id' in numeric_columns:
            numeric_columns.remove('sample_id')
        
        X = df[numeric_columns].values
        data_name = results['data_name']
        
        os.makedirs(output_dir, exist_ok=True)
        
        silhouette_scores = {'kmeans': [], 'kmedoids': []}
        
        for algorithm in results['algorithms']:
            for k in results['k_values']:
                labels = np.array(results['metrics'][algorithm]['labels'][str(k)])
                
                if len(labels) > 0 and len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    silhouette_scores[algorithm].append(score)
                else:
                    silhouette_scores[algorithm].append(0)
        
        # Plotar scores de silhouette
        fig = go.Figure()
        
        for algorithm in results['algorithms']:
            fig.add_trace(go.Scatter(
                x=results['k_values'],
                y=silhouette_scores[algorithm],
                mode='lines+markers',
                name=f'{algorithm.upper()} - Silhouette',
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f'Análise de Silhouette - {data_name}',
            xaxis_title='Número de Clusters (K)',
            yaxis_title='Score de Silhouette',
            template='plotly_white',
            height=500
        )
        
        silhouette_path = os.path.join(output_dir, f'silhouette_analysis_{data_name.lower().replace(" ", "_")}.html')
        fig.write_html(silhouette_path)
        print(f"Análise de silhouette salva em: {silhouette_path}")
        
    except ImportError:
        print("Silhouette analysis requer scikit-learn. Pulando esta análise.")
    except Exception as e:
        print(f"Erro na análise de silhouette: {e}")
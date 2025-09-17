"""
PCA reduction module.
Responsável por rodar PCA para 90%, 80% e 75% da variância,
salvar resultados em CSV e gráficos da variância.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from tqdm import tqdm


def apply_pca_by_variance(X, variance_ratios=[0.90, 0.80, 0.75], random_state=42):
    """
    Aplica PCA para diferentes níveis de variância explicada.
    
    Args:
        X (np.array): Dados de entrada
        variance_ratios (list): Lista de porcentagens de variância a manter
        random_state (int): Seed para reprodutibilidade
    
    Returns:
        dict: Resultados para cada nível de variância
    """
    print(f"Aplicando PCA em dados: {X.shape}")
    
    results = {}
    
    for var_ratio in variance_ratios:
        print(f"\n=== PCA para {var_ratio*100:.0f}% da variância ===")
        
        # Aplicar PCA
        pca = PCA(n_components=var_ratio, random_state=random_state)
        X_pca = pca.fit_transform(X)
        
        n_components = pca.n_components_
        explained_variance = pca.explained_variance_ratio_.sum()
        
        print(f"Componentes necessárias: {n_components}")
        print(f"Variância explicada real: {explained_variance:.4f}")
        print(f"Shape dos dados transformados: {X_pca.shape}")
        
        results[var_ratio] = {
            'model': pca,
            'transformed_data': X_pca,
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance_total': explained_variance,
            'singular_values': pca.singular_values_,
            'components': pca.components_
        }
    
    return results


def save_pca_results(pca_results, image_paths, output_dir="outputs/embeddings"):
    """
    Salva resultados do PCA em CSV com nomes conforme enunciado.
    
    Args:
        pca_results (dict): Resultados do PCA
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
    
    Returns:
        dict: Caminhos dos arquivos salvos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    for var_ratio, result in pca_results.items():
        X_pca = result['transformed_data']
        n_components = result['n_components']
        
        # Criar DataFrame com componentes principais
        columns = [f'PC{i+1}' for i in range(n_components)]
        df = pd.DataFrame(X_pca, columns=columns)
        
        # Adicionar informações das amostras
        df['sample_id'] = range(len(df))
        df['image_path'] = image_paths
        df['folder'] = [os.path.basename(os.path.dirname(path)) for path in image_paths]
        df['filename'] = [os.path.basename(path) for path in image_paths]
        
        # Salvar CSV com nome conforme enunciado (pca_var90.csv, etc.)
        filename = f"pca_var{int(var_ratio*100)}.csv"
        csv_path = os.path.join(output_dir, filename)
        df.to_csv(csv_path, index=False)
        
        print(f"PCA {var_ratio*100:.0f}% variância salvo em: {csv_path}")
        print(f"  → {n_components} componentes resultantes")
        saved_files[var_ratio] = csv_path
    
    return saved_files


def create_variance_explained_plot(pca_results, save_path):
    """
    Cria gráfico da variância explicada por componente.
    
    Args:
        pca_results (dict): Resultados do PCA
        save_path (str): Caminho para salvar figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (var_ratio, result) in enumerate(pca_results.items()):
        explained_var = result['explained_variance_ratio']
        cumulative_var = np.cumsum(explained_var)
        
        # Plot individual
        ax = axes[idx] if idx < len(axes) else axes[-1]
        
        # Gráfico de barras da variância individual
        ax.bar(range(1, len(explained_var) + 1), explained_var, 
               alpha=0.7, color=colors[idx % len(colors)], 
               label=f'Individual ({var_ratio*100:.0f}%)')
        
        # Linha da variância cumulativa
        ax2 = ax.twinx()
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                'ko-', alpha=0.8, label='Cumulativa')
        ax2.set_ylabel('Variância Cumulativa')
        ax2.set_ylim(0, 1)
        
        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Variância Explicada')
        ax.set_title(f'PCA - {var_ratio*100:.0f}% Variância\\n{result["n_components"]} componentes')
        ax.legend(loc='upper right')
        ax2.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Remover subplot extra se houver
    if len(pca_results) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de variância explicada salvo em: {save_path}")


def create_pca_comparison_plot(pca_results, save_path):
    """
    Cria gráfico comparativo dos primeiros componentes.
    
    Args:
        pca_results (dict): Resultados do PCA
        save_path (str): Caminho para salvar figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (var_ratio, result) in enumerate(pca_results.items()):
        X_pca = result['transformed_data']
        
        if idx < len(axes):
            ax = axes[idx]
            
            # Scatter plot dos primeiros 2 componentes
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               alpha=0.6, c=colors[idx % len(colors)], s=30)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'PCA {var_ratio*100:.0f}% - PC1 vs PC2\\n'
                        f'{result["n_components"]} componentes')
            ax.grid(True, alpha=0.3)
    
    # Remover subplot extra se houver
    if len(pca_results) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico comparativo PCA salvo em: {save_path}")


def create_interactive_pca_plot(pca_results, image_paths, save_path):
    """
    Cria visualizações interativas do PCA com Plotly.
    
    Args:
        pca_results (dict): Resultados do PCA
        image_paths (list): Caminhos das imagens
        save_path (str): Caminho para salvar figura HTML
    """
    # Preparar dados para todas as variâncias
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    filenames = [os.path.basename(path) for path in image_paths]
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (var_ratio, result) in enumerate(pca_results.items()):
        X_pca = result['transformed_data']
        
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            name=f'PCA {var_ratio*100:.0f}% ({result["n_components"]} comp.)',
            marker=dict(
                color=colors[idx % len(colors)],
                size=6,
                opacity=0.7
            ),
            text=[f'File: {f}<br>Folder: {l}' for f, l in zip(filenames, labels)],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='PCA Comparison - PC1 vs PC2',
        xaxis_title='PC1',
        yaxis_title='PC2',
        hovermode='closest',
        width=800,
        height=600
    )
    
    fig.write_html(save_path)
    print(f"Visualização interativa PCA salva em: {save_path}")


def analyze_pca_components(pca_results, save_path):
    """
    Analisa e salva informações detalhadas dos componentes PCA.
    
    Args:
        pca_results (dict): Resultados do PCA
        save_path (str): Caminho para salvar análise
    """
    analysis = {}
    
    for var_ratio, result in pca_results.items():
        model = result['model']
        
        analysis[f'pca_{int(var_ratio*100)}percent'] = {
            'n_components': int(result['n_components']),  # Converter para int Python
            'explained_variance_ratio_total': float(result['explained_variance_total']),
            'explained_variance_ratio_per_component': [float(x) for x in result['explained_variance_ratio']],
            'singular_values': [float(x) for x in result['singular_values']],
            'component_importance': {
                f'PC{i+1}': float(var) 
                for i, var in enumerate(result['explained_variance_ratio'])
            }
        }
    
    # Salvar como JSON
    import json
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Análise detalhada PCA salva em: {save_path}")


def run_pca_analysis(X, image_paths, output_dir="outputs", 
                    variance_ratios=[0.90, 0.80, 0.75]):
    """
    Executa análise completa do PCA.
    
    Args:
        X (np.array): Dados de entrada
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
        variance_ratios (list): Níveis de variância para analisar
    
    Returns:
        dict: Resultados completos da análise PCA
    """
    print("=== Iniciando Análise PCA ===")
    
    # Aplicar PCA
    pca_results = apply_pca_by_variance(X, variance_ratios)
    
    # Salvar embeddings
    saved_csvs = save_pca_results(pca_results, image_paths, 
                                 os.path.join(output_dir, "embeddings"))
    
    # Salvar modelos PCA
    for var_ratio, result in pca_results.items():
        model_path = os.path.join(output_dir, "embeddings", 
                                 f"pca_{int(var_ratio*100)}percent_model.joblib")
        joblib.dump(result['model'], model_path)
        print(f"Modelo PCA {var_ratio*100:.0f}% salvo em: {model_path}")
    
    # Criar visualizações
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Gráfico de variância explicada
    variance_plot_path = os.path.join(figures_dir, "pca_variance_explained.png")
    create_variance_explained_plot(pca_results, variance_plot_path)
    
    # Gráfico comparativo
    comparison_plot_path = os.path.join(figures_dir, "pca_comparison.png")
    create_pca_comparison_plot(pca_results, comparison_plot_path)
    
    # Visualização interativa
    interactive_plot_path = os.path.join(figures_dir, "pca_interactive.html")
    create_interactive_pca_plot(pca_results, image_paths, interactive_plot_path)
    
    # Análise detalhada
    analysis_path = os.path.join(output_dir, "embeddings", "pca_analysis.json")
    analyze_pca_components(pca_results, analysis_path)
    
    print("\n=== Análise PCA Concluída ===")
    print(f"Resultados salvos para {len(variance_ratios)} níveis de variância")
    
    return {
        'pca_results': pca_results,
        'saved_csvs': saved_csvs,
        'figures': {
            'variance_plot': variance_plot_path,
            'comparison_plot': comparison_plot_path,
            'interactive_plot': interactive_plot_path
        },
        'analysis': analysis_path
    }


if __name__ == "__main__":
    # Exemplo de uso
    from preprocess import load_preprocessed_data
    
    # Carregar dados preprocessados
    processed_path = "data/processed/preprocessed_data.joblib"
    if os.path.exists(processed_path):
        data = load_preprocessed_data(processed_path)
        X_scaled = data['X_scaled']
        image_paths = data['image_paths']
        
        # Executar análise PCA
        print("Executando análise PCA...")
        results = run_pca_analysis(
            X_scaled, 
            image_paths,
            variance_ratios=[0.95, 0.90, 0.80, 0.75]  # Incluir 95% também
        )
        
        print("\nAnálise PCA concluída!")
        
    else:
        print("Execute primeiro preprocess.py para preprocessar os dados!")
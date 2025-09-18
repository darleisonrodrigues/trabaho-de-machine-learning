"""
Script principal para executar a análise de clustering (Etapa 1.2).
"""

import os
import sys
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.clustering import (
    apply_clustering_analysis,
    create_clustering_visualizations,
    save_clustering_results,
    print_clustering_summary
)
from src.visualize_clustering import (
    create_comprehensive_clustering_dashboard,
    create_cluster_distribution_analysis,
    create_silhouette_analysis
)
from config import OUTPUTS_DIR, RANDOM_STATE


def step_7_clustering_analysis():
    """
    Etapa 1.2: Análise de Clustering com K-means e K-medoids
    """
    print("\n" + "="*60)
    print("ETAPA 1.2 - ALGORITMOS DE CLUSTERIZAÇÃO")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Definir valores de K para testar
    # Dataset tem 20 pessoas, então testamos de 2 até 25 clusters
    # Faixa prioritária: K=15-25 (próximo ao número real de identidades)
    k_values = list(range(2, 26))  # K = 2, 3, 4, ..., 25
    print(f"Valores de K a testar: {k_values}")
    print(f"Faixa prioritária para análise: K=15-25 (dataset com ~20 identidades)")
    
    # Definir embeddings para analisar (removido t-SNE - apenas para visualização)
    embeddings_to_analyze = [
        {
            'path': 'outputs/embeddings/pca_var90.csv',
            'name': 'PCA 90%',
            'description': '59 dimensões (90% da variância)'
        },
        {
            'path': 'outputs/embeddings/pca_var80.csv',
            'name': 'PCA 80%',
            'description': '31 dimensões (80% da variância)'
        },
        {
            'path': 'outputs/embeddings/pca_var75.csv',
            'name': 'PCA 75%',
            'description': '25 dimensões (75% da variância)'
        },
        {
            'path': 'outputs/embeddings/umap_3.csv',
            'name': 'UMAP 3D',
            'description': '3 dimensões (UMAP target_dim=3)'
        },
        {
            'path': 'outputs/embeddings/umap_15.csv',
            'name': 'UMAP 15D',
            'description': '15 dimensões (UMAP target_dim=15)'
        },
        {
            'path': 'outputs/embeddings/umap_55.csv',
            'name': 'UMAP 55D',
            'description': '55 dimensões (UMAP target_dim=55)'
        },
        {
            'path': 'outputs/embeddings/umap_101.csv',
            'name': 'UMAP 101D',
            'description': '101 dimensões (UMAP target_dim=101)'
        }
    ]
    
    print(f"\nEmbeddings selecionados para clustering:")
    for emb in embeddings_to_analyze:
        print(f"  - {emb['name']}: {emb['description']}")
    
    # Criar diretórios de saída
    clustering_dir = os.path.join(OUTPUTS_DIR, 'clustering')
    results_dir = os.path.join(clustering_dir, 'results')
    figures_dir = os.path.join(clustering_dir, 'figures')
    
    for dir_path in [clustering_dir, results_dir, figures_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Executar análise de clustering para cada embedding
    all_results = []
    
    for embedding_info in embeddings_to_analyze:
        data_path = embedding_info['path']
        data_name = embedding_info['name']
        
        print(f"\n{'-'*40}")
        print(f"Processando: {data_name}")
        print(f"Arquivo: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"ERRO: Arquivo não encontrado: {data_path}")
            continue
        
        try:
            # Aplicar clustering
            results = apply_clustering_analysis(
                data_path=data_path,
                data_name=data_name,
                k_values=k_values,
                algorithms=['kmeans', 'kmedoids']
            )
            
            all_results.append(results)
            
            # Salvar resultados individuais
            result_filename = f"clustering_results_{data_name.lower().replace(' ', '_').replace('%', 'pct')}.json"
            result_path = os.path.join(results_dir, result_filename)
            save_clustering_results(results, result_path)
            
            # Criar visualizações individuais
            embedding_figures_dir = os.path.join(figures_dir, data_name.lower().replace(' ', '_').replace('%', 'pct'))
            create_clustering_visualizations(data_path, results, embedding_figures_dir)
            
            # Análises adicionais
            create_cluster_distribution_analysis(data_path, results, embedding_figures_dir)
            create_silhouette_analysis(data_path, results, embedding_figures_dir)
            
            # Imprimir resumo
            print_clustering_summary(results)
            
        except Exception as e:
            print(f"ERRO ao processar {data_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Criar dashboard comparativo
    if all_results:
        print(f"\n{'-'*40}")
        print("Criando dashboard comparativo...")
        
        dashboard_dir = os.path.join(figures_dir, 'dashboard')
        create_comprehensive_clustering_dashboard(all_results, dashboard_dir)
        
        # Salvar resultados consolidados
        consolidated_path = os.path.join(results_dir, 'consolidated_clustering_results.json')
        consolidated_results = {
            'timestamp': datetime.now().isoformat(),
            'k_values': k_values,
            'embeddings_analyzed': len(all_results),
            'results': all_results
        }
        
        import json
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados consolidados salvos em: {consolidated_path}")
        
        # Imprimir análise final
        print_final_analysis(all_results)
    
    else:
        print("ERRO: Nenhum resultado de clustering foi obtido!")
    
    print(f"\n{'='*60}")
    print("ETAPA 1.2 CONCLUÍDA!")
    print(f"Resultados salvos em: {clustering_dir}")
    print(f"{'='*60}")


def print_final_analysis(all_results):
    """Imprime análise final comparativa dos resultados."""
    
    print(f"\n{'='*60}")
    print("ANÁLISE FINAL COMPARATIVA")
    print(f"{'='*60}")
    
    # Encontrar melhores resultados globais
    best_overall = {
        'kmeans': {'dunn': 0, 'embedding': '', 'k': 0},
        'kmedoids': {'dunn': 0, 'embedding': '', 'k': 0}
    }
    
    for results in all_results:
        for algorithm in ['kmeans', 'kmedoids']:
            if algorithm in results['algorithms']:
                dunn_values = results['metrics'][algorithm]['dunn_index']
                if max(dunn_values) > best_overall[algorithm]['dunn']:
                    best_idx = dunn_values.index(max(dunn_values))
                    best_overall[algorithm] = {
                        'dunn': max(dunn_values),
                        'embedding': results['data_name'],
                        'k': results['k_values'][best_idx]
                    }
    
    print("\n🏆 MELHORES RESULTADOS GLOBAIS:")
    for algorithm, best in best_overall.items():
        if best['dunn'] > 0:
            print(f"{algorithm.upper()}: {best['embedding']} com K={best['k']} (Dunn: {best['dunn']:.4f})")
    
    # Análise por embedding
    print(f"\n📊 ANÁLISE POR EMBEDDING:")
    for results in all_results:
        print(f"\n{results['data_name']} ({results['data_shape'][1]} dimensões):")
        
        for algorithm in ['kmeans', 'kmedoids']:
            if algorithm in results['algorithms']:
                dunn_values = results['metrics'][algorithm]['dunn_index']
                if max(dunn_values) > 0:
                    best_idx = dunn_values.index(max(dunn_values))
                    best_k = results['k_values'][best_idx]
                    best_dunn = dunn_values[best_idx]
                    print(f"  {algorithm}: K={best_k}, Dunn={best_dunn:.4f}")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    
    # Melhor embedding por dimensionalidade
    low_dim_results = [r for r in all_results if r['data_shape'][1] <= 10]
    high_dim_results = [r for r in all_results if r['data_shape'][1] > 10]
    
    if low_dim_results:
        best_low_dim = max(low_dim_results, 
                          key=lambda x: max([max(x['metrics'][alg]['dunn_index']) 
                                           for alg in x['algorithms']]))
        print(f"• Para baixa dimensionalidade: {best_low_dim['data_name']}")
    
    if high_dim_results:
        best_high_dim = max(high_dim_results, 
                           key=lambda x: max([max(x['metrics'][alg]['dunn_index']) 
                                            for alg in x['algorithms']]))
        print(f"• Para alta dimensionalidade: {best_high_dim['data_name']}")
    
    # Avaliação dos algoritmos
    kmeans_scores = []
    kmedoids_scores = []
    
    for results in all_results:
        if 'kmeans' in results['algorithms']:
            kmeans_scores.extend(results['metrics']['kmeans']['dunn_index'])
        if 'kmedoids' in results['algorithms']:
            kmedoids_scores.extend(results['metrics']['kmedoids']['dunn_index'])
    
    if kmeans_scores and kmedoids_scores:
        avg_kmeans = sum(kmeans_scores) / len(kmeans_scores)
        avg_kmedoids = sum(kmedoids_scores) / len(kmedoids_scores)
        
        better_algorithm = "K-means" if avg_kmeans > avg_kmedoids else "K-medoids"
        print(f"• Algoritmo com melhor performance média: {better_algorithm}")
        print(f"  (K-means: {avg_kmeans:.4f}, K-medoids: {avg_kmedoids:.4f})")


if __name__ == "__main__":
    step_7_clustering_analysis()
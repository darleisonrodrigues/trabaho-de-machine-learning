"""
Script principal para execução do pipeline completo de Redução de Dimensionalidade.
Executa todo o fluxo: carregamento, preprocessamento, PCA, t-SNE, UMAP e visualizações.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import load_images_from_folder, save_dataset, load_dataset
from preprocess import preprocess_and_save_data, load_preprocessed_data
from reduce_pca import run_pca_analysis
from reduce_tsne import run_tsne_analysis
from reduce_umap import run_umap_analysis
from visualize import create_embedding_dashboard


def print_header():
    """Imprime cabeçalho do programa."""
    print("=" * 80)
    print("🔬 ANÁLISE DE REDUÇÃO DE DIMENSIONALIDADE - MACHINE LEARNING")
    print("=" * 80)
    print("📁 Projeto para análise de imagens de faces com:")
    print("   • PCA (90%, 80%, 75% da variância)")
    print("   • t-SNE (visualização 2D)")
    print("   • UMAP (3, 15, 55, 101 dimensões)")
    print("=" * 80)
    print()


def check_data_directory():
    """Verifica se o diretório de dados existe."""
    data_dir = "data/raw/RecFac"
    if not os.path.exists(data_dir):
        print(f"❌ ERRO: Diretório {data_dir} não encontrado!")
        print(f"📝 Por favor, coloque as imagens do dataset em: {os.path.abspath(data_dir)}")
        print("   Estrutura esperada:")
        print("   data/raw/RecFac/")
        print("   ├── pessoa1/")
        print("   │   ├── img1.png")
        print("   │   └── img2.png")
        print("   ├── pessoa2/")
        print("   │   ├── img1.png")
        print("   │   └── img2.png")
        print("   └── ...")
        return False
    return True


def step_1_load_dataset():
    """Passo 1: Carregar e processar dataset."""
    print("🚀 PASSO 1: Carregamento do Dataset")
    print("-" * 50)
    
    raw_data_path = "data/raw/RecFac"
    processed_data_path = "data/processed/dataset.joblib"
    
    if os.path.exists(processed_data_path):
        print("📂 Dataset já processado encontrado. Carregando...")
        X, image_paths = load_dataset(processed_data_path)
    else:
        print("📷 Carregando e processando imagens...")
        X, image_paths = load_images_from_folder(raw_data_path)
        save_dataset(X, image_paths, processed_data_path)
    
    print(f"✅ Dataset carregado: {X.shape[0]} imagens de {X.shape[1]} features")
    print()
    
    return X, image_paths


def step_2_preprocess_data(X, image_paths):
    """Passo 2: Preprocessar dados."""
    print("🔧 PASSO 2: Preprocessamento dos Dados")
    print("-" * 50)
    
    processed_path = "data/processed/preprocessed_data.joblib"
    
    if os.path.exists(processed_path):
        print("📊 Dados preprocessados encontrados. Carregando...")
        data = load_preprocessed_data(processed_path)
        X_scaled = data['X_scaled']
    else:
        print("⚙️ Aplicando StandardScaler (z-score)...")
        X_scaled, _ = preprocess_and_save_data(X, image_paths)
    
    print(f"✅ Dados padronizados: média={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
    print()
    
    return X_scaled


def step_3_pca_analysis(X_scaled, image_paths):
    """Passo 3: Análise PCA."""
    print("📊 PASSO 3: Análise PCA")
    print("-" * 50)
    
    print("🔍 Executando PCA para diferentes níveis de variância...")
    
    pca_results = run_pca_analysis(
        X_scaled, 
        image_paths,
        variance_ratios=[0.90, 0.80, 0.75]  # Exatamente conforme enunciado
    )
    
    print("✅ Análise PCA concluída!")
    
    # Resumo dos resultados
    for var_ratio, result in pca_results['pca_results'].items():
        n_comp = result['n_components']
        var_explained = result['explained_variance_total']
        print(f"   • {var_ratio*100:.0f}% variância: {n_comp} componentes (real: {var_explained:.3f})")
    
    print()
    return pca_results


def step_4_tsne_analysis(X_scaled, image_paths):
    """Passo 4: Análise t-SNE."""
    print("🎯 PASSO 4: Análise t-SNE")
    print("-" * 50)
    
    print("🔮 Executando t-SNE para visualização 2D...")
    
    tsne_results = run_tsne_analysis(
        X_scaled, 
        image_paths,
        perplexity_values=[30],  # Foco no perplexity=30 como principal
        use_pca_first=True
    )
    
    print("✅ Análise t-SNE concluída!")
    print(f"   • {len(tsne_results)} configurações de perplexity testadas")
    print()
    
    return tsne_results


def step_5_umap_analysis(X_scaled, image_paths):
    """Passo 5: Análise UMAP."""
    print("🗺️ PASSO 5: Análise UMAP")
    print("-" * 50)
    
    print("🚀 Executando UMAP para diferentes dimensões...")
    
    umap_results = run_umap_analysis(
        X_scaled, 
        image_paths,
        n_components_list=[3, 15, 55, 101],  # Exatamente conforme enunciado
        n_neighbors=15,
        min_dist=0.1
    )
    
    print("✅ Análise UMAP concluída!")
    print(f"   • Embeddings criados para: {list(umap_results['umap_results'].keys())} dimensões")
    print()
    
    return umap_results


def step_6_create_dashboard(X_scaled, image_paths, pca_results, tsne_results, umap_results):
    """Passo 6: Criar dashboard de visualizações."""
    print("📈 PASSO 6: Criando Dashboard de Visualizações")
    print("-" * 50)
    
    # Preparar embeddings para dashboard
    embeddings_dict = {}
    
    # Adicionar PCA
    for var_ratio, result in pca_results['pca_results'].items():
        name = f"PCA {var_ratio*100:.0f}%"
        embeddings_dict[name] = result['transformed_data']
    
    # Adicionar t-SNE (primeiro resultado)
    if tsne_results:
        first_tsne = list(tsne_results.values())[0]
        embeddings_dict["t-SNE"] = first_tsne['embedding']
    
    # Adicionar UMAP (primeiros resultados)
    for n_comp in [3, 15]:  # Apenas alguns para o dashboard
        if n_comp in umap_results['umap_results']:
            name = f"UMAP {n_comp}D"
            embeddings_dict[name] = umap_results['umap_results'][n_comp]['embedding']
    
    # Criar dashboard
    print("🎨 Gerando visualizações comparativas...")
    dashboard_files = create_embedding_dashboard(embeddings_dict, image_paths)
    
    print("✅ Dashboard criado!")
    print(f"   • {len(dashboard_files)} arquivos de visualização gerados")
    print()
    
    return dashboard_files


def print_summary():
    """Imprime resumo dos resultados."""
    print("📋 RESUMO DOS RESULTADOS")
    print("-" * 50)
    
    outputs_dir = "outputs"
    
    # Listar arquivos de embeddings
    embeddings_dir = os.path.join(outputs_dir, "embeddings")
    if os.path.exists(embeddings_dir):
        csv_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.csv')]
        print(f"📊 Embeddings salvos: {len(csv_files)} arquivos CSV")
        for file in sorted(csv_files):
            print(f"   • {file}")
    
    print()
    
    # Listar figuras
    figures_dir = os.path.join(outputs_dir, "figures")
    if os.path.exists(figures_dir):
        png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        html_files = [f for f in os.listdir(figures_dir) if f.endswith('.html')]
        print(f"📈 Visualizações criadas: {len(png_files)} PNG + {len(html_files)} HTML")
        
        print("   Gráficos estáticos (.png):")
        for file in sorted(png_files):
            print(f"   • {file}")
        
        print("   Gráficos interativos (.html):")
        for file in sorted(html_files):
            print(f"   • {file}")
    
    print()
    print("🎯 PRÓXIMOS PASSOS:")
    print("   1. Analise os gráficos em outputs/figures/")
    print("   2. Use os embeddings CSV para clustering (K-means, K-medoids)")
    print("   3. Compare a separabilidade visual entre os métodos")
    print("   4. Escolha os melhores parâmetros para seu projeto final")
    print()


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Pipeline de Redução de Dimensionalidade')
    parser.add_argument('--skip-check', action='store_true', 
                       help='Pular verificação do diretório de dados')
    args = parser.parse_args()
    
    # Começar cronômetro
    start_time = time.time()
    
    try:
        # Cabeçalho
        print_header()
        
        # Verificar diretório de dados
        if not args.skip_check and not check_data_directory():
            return 1
        
        # Pipeline completo
        print("🔄 INICIANDO PIPELINE COMPLETO")
        print("=" * 80)
        print()
        
        # Passo 1: Carregar dataset
        X, image_paths = step_1_load_dataset()
        
        # Passo 2: Preprocessar
        X_scaled = step_2_preprocess_data(X, image_paths)
        
        # Passo 3: PCA
        pca_results = step_3_pca_analysis(X_scaled, image_paths)
        
        # Passo 4: t-SNE
        tsne_results = step_4_tsne_analysis(X_scaled, image_paths)
        
        # Passo 5: UMAP
        umap_results = step_5_umap_analysis(X_scaled, image_paths)
        
        # Passo 6: Dashboard
        dashboard_files = step_6_create_dashboard(
            X_scaled, image_paths, pca_results, tsne_results, umap_results
        )
        
        # Tempo total
        total_time = time.time() - start_time
        
        # Resumo final
        print("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 80)
        print(f"⏱️ Tempo total: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
        print()
        
        print_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ ERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
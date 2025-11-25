"""
RetailX Customer Segmentation (Versão 4.0 - Análise de Granularidade K=5)

Objetivo: Reexecutar a clusterização utilizando K=5 (o segundo melhor k, para maior
          granularidade de negócio), mantendo as correções de pré-processamento
          (Log-Transformação e exclusão de binárias do StandardScaler).

Modificação Principal: A variável 'best_k' é sobrescrita para K=5.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json

# ===================================================
# CONFIGURAÇÃO DE DIRETÓRIOS
# ===================================================
DATA_DIR = Path('C:/Repos/Python/Clusters')
OUT_DIR = DATA_DIR

# ------------------------------
# Função segura para ler arquivos
# ------------------------------
def safe_read(fname):
    p = DATA_DIR / fname
    if not p.exists():
        print(f'File not found: {p}')
        return None
    try:
        df = pd.read_csv(p, parse_dates=True) 
        print(f'Loaded {fname} shape={df.shape}')
        return df
    except Exception as e:
        print(f'Error loading {fname}:', e)
        return None


# ------------------------------
# Função para gerar EDA simples
# ------------------------------
def summarize_df(df, name):
    print("\n" + "="*40)
    print(f"Summary for {name}:")
    print(df.head(3).to_string())
    print("\nInfo:")
    df.info(verbose=False, memory_usage="deep")
    print("\nDescribe numeric:")
    print(df.describe().T)


# ------------------------------
# Carregar arquivos enviados
# ------------------------------
customer = safe_read('clustering_customer.csv')
product = safe_read('clustering_product.csv')
payment = safe_read('clustering_payment.csv')
features = safe_read('clustering_features.csv')

# ===================================================
# 1) Construção do DataFrame Principal
# ===================================================

df = None
customer_cols_meta = ['customer_id', 'age', 'hh_income', 'omni_shopper', 'email_subscribed']

if features is not None and 'customer_id' in features.columns:
    df = features.copy()
    if customer is not None and 'customer_id' in customer.columns:
        df = df.merge(customer[[c for c in customer_cols_meta if c in customer.columns]],
                      on='customer_id', how='left')
    print("\nUsing clustering_features.csv as main features table.")

else:
    print("\nBuilding features from scratch using payment and product tables (if available).")
    dfs = []
    
    if customer is not None and 'customer_id' in customer.columns: dfs.append(customer.copy())
        
    # --- Lógica para construir features de payment ---
    if payment is not None:
        pay = payment.copy()
        cid, datecol, amtcol = None, None, None
        for c in ['customer_id','cust_id','client_id','id_customer','buyer_id']:
            if c in pay.columns: cid = c; break
        for c in pay.columns:
            if 'date' in c.lower() or 'purchase' in c.lower(): datecol = c; break
        for c in pay.columns:
            if 'amount' in c.lower() or 'total' in c.lower() or 'value' in c.lower() or 'price' in c.lower(): amtcol = c; break

        if cid is not None:
            if datecol: pay[datecol] = pd.to_datetime(pay[datecol], errors='coerce')
            agg = pay.groupby(cid).agg(
                frequency=(datecol, lambda x: x.dropna().nunique() if datecol else len(x)),
                monetary=(amtcol, 'sum') if amtcol else (cid, 'count'),
                avg_ticket=(amtcol, 'mean') if amtcol else (cid, 'count'),
                last_purchase=(datecol, 'max') if datecol else (cid, 'count')
            ).reset_index().rename(columns={cid: 'customer_id'})
            dfs.append(agg)
            
    # --- Lógica para construir features de produto ---
    if product is not None:
        prod = product.copy()
        cid, catcol = None, None
        for c in ['customer_id','cust_id','client_id','id_customer','buyer_id']:
            if c in prod.columns: cid = c; break
        for c in prod.columns:
            if 'category' in c.lower(): catcol = c; break

        if cid:
            if catcol:
                pagg = prod.groupby(cid).agg(
                    distinct_categories=(catcol, 'nunique')
                ).reset_index().rename(columns={cid: 'customer_id'})
            else:
                pagg = prod.groupby(cid).size().reset_index(
                    name='products_bought'
                ).rename(columns={cid: 'customer_id'})
            dfs.append(pagg)

    if dfs:
        df = dfs[0]
        for other in dfs[1:]:
            df = df.merge(other, on='customer_id', how='outer')
    else:
        raise SystemExit("Unable to construct feature table.")

if df is None:
    raise SystemExit("Final DataFrame (df) is empty. Check input files.")

# --- Criação da Recência e preenchimento de NaNs ---
if 'last_purchase' in df.columns:
    df['last_purchase'] = pd.to_datetime(df['last_purchase'], errors='coerce')
    reference_date = df['last_purchase'].max(skipna=True)
    if pd.notna(reference_date):
        reference_date = reference_date + pd.Timedelta(days=1)
        df['recency_days'] = (reference_date - df['last_purchase']).dt.days

if 'monetary' not in df.columns and 'avg_ticket' in df.columns and 'frequency' in df.columns:
    df['monetary'] = df['avg_ticket'] * df['frequency']

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)


# ===================================================
# 2) Seleção de Features e Pré-processamento
# ===================================================

# Variáveis binárias a serem EXCLUÍDAS do clustering (corrigindo o problema do peso)
binary_cols_to_exclude = ['omni_shopper', 'email_subscribed'] 

# Lista completa de features contínuas/RFM/demográficas
all_candidate_features = [
    'recency_days', 'frequency', 'monetary', 'avg_ticket', 
    'distinct_categories', 'products_bought', 'sales', 'units', 'orders',
    'age', 'hh_income'
]

# 1. Montar a lista final de features CONTÍNUAS para clusterização
features_for_clustering = [c for c in all_candidate_features if c in df.columns]
features_for_clustering = [c for c in features_for_clustering if c not in binary_cols_to_exclude]

if len(features_for_clustering) < 2:
    raise SystemExit("Not enough continuous features remaining for clustering.")

print("\nFeatures contínuas/RFM para Clustering (após exclusão de binárias):", features_for_clustering)

# Criar X para clusterização
X = df[features_for_clustering].copy().astype(float)

# --- Log-Transformação ---
log_transform_cols = ['monetary', 'avg_ticket', 'sales', 'units', 'orders', 'hh_income']
original_features_map = {}

for col in log_transform_cols:
    if col in X.columns:
        X[f'log_{col}'] = np.log1p(X[col])
        original_features_map[f'log_{col}'] = col 
        X = X.drop(columns=[col])

print(f"Features transformadas para log: {[v for k, v in original_features_map.items()]}")

# ===================================================
# 3) Normalizar dados e Clusterizar
# ===================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_columns_final = X.columns.tolist() # Nomes das colunas normalizadas (log_...)

# --- Seleção do K Ótimo (Mantida para fins de relatório) ---
sil_scores = {}
max_k = min(10, len(X) - 1)
for k in range(2, max_k + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    try:
        score = silhouette_score(X_scaled, labels)
        sil_scores[k] = score
        print(f"k={k}, silhouette={sil_scores[k]:.4f}")
    except ValueError:
        pass
        
best_k_auto = max(sil_scores, key=sil_scores.get) if sil_scores else 3 
print(f"\nOptimal k (via Silhouette): {best_k_auto}")

# ====================================================================
# [MODIFICAÇÃO CRÍTICA]: Forçar K para análise de granularidade K=5
# ====================================================================
K_FORCED = 5
if K_FORCED in sil_scores:
    print(f"\n✅ Override: Usando K={K_FORCED} para análise de granularidade (Silhouette: {sil_scores[K_FORCED]:.4f})")
else:
    print(f"\n⚠️ Aviso: K={K_FORCED} não estava na lista de scores. Executando KMeans com K={K_FORCED}...")
    
best_k = K_FORCED # Define K final para o KMeans


# Rodar KMeans final
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA para visualização 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'], df['pca2'] = X_pca[:,0], X_pca[:,1]


# ===================================================
# 4) Exportar resultados e Resumo Tabular
# ===================================================
df.to_csv(OUT_DIR / "clustered_customers_k5.csv", index=False) # Mudando o nome do arquivo para refletir K=5

# Gráficos
plt.figure(figsize=(6,4))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'K={best_k} (Forçado)')
plt.title("Silhouette Score by k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "silhouette_scores.png")
plt.close()

plt.figure(figsize=(6,5))
plt.scatter(df['pca1'], df['pca2'], c=df['cluster'])
plt.title(f"PCA - Customer Clusters (K={best_k} Forçado)")
plt.xlabel("pca1")
plt.ylabel("pca2")
plt.savefig(OUT_DIR / "pca_scatter_k5.png") # Mudando o nome do arquivo para refletir K=5
plt.close()

# --- Gerar Resumo Tabular ---
summary_list = []
report_cols = [c for c in customer_cols_meta if c in df.columns and c != 'customer_id']
report_cols.extend([c for c in features_for_clustering if c not in report_cols])

for c in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == c]
    
    stats = {
        "cluster": int(c),
        "count": int(len(subset)),
        "%_of_total": round(len(subset) / len(df) * 100, 1)
    }
    
    mean_stats = subset[report_cols].mean().round(2).to_dict()
    
    for key, value in mean_stats.items():
        if key in binary_cols_to_exclude:
            stats[f'%_{key}_true'] = value * 100
        else:
            stats[f'AVG_{key}'] = value

    summary_list.append(stats)

# Criar o DataFrame de Resumo
summary_df = pd.DataFrame(summary_list)

# Renomear e reordenar colunas
final_cols_order = ['cluster', 'count', '%_of_total']
demographic_cols = ['age', 'hh_income']
for col in demographic_cols:
    if f'AVG_{col}' in summary_df.columns: final_cols_order.append(f'AVG_{col}')

for col in binary_cols_to_exclude:
    if f'%_{col}_true' in summary_df.columns: final_cols_order.append(f'%_{col}_true')

for col in summary_df.columns:
    if col.startswith('AVG_') and col.replace('AVG_', '') not in demographic_cols:
        final_cols_order.append(col)

summary_df_business = summary_df[
    [c for c in final_cols_order if c in summary_df.columns]
]

# Exportar o resumo tabular
summary_df_business.to_csv(OUT_DIR / "cluster_summary_table_k5.csv", index=False) # Mudando o nome do arquivo

# Criar resumo JSON
cluster_summary = {
    "K_USED": K_FORCED,
    "features_used_clustering": X_columns_final,
    "features_log_transformed_original": [v for k, v in original_features_map.items()],
    "features_excluded_binary": binary_cols_to_exclude,
    "best_k_auto_silhouette": int(best_k_auto),
    "silhouette_scores": sil_scores,
    "cluster_report_table": summary_df.to_dict(orient='records')
}

with open(OUT_DIR / "cluster_summary_k5.json", "w") as f:
    json.dump(cluster_summary, f, indent=2) # Mudando o nome do arquivo

print("\n--- Resultados K=5 Gerados ---")
print(f"K final usado: {best_k}")
print("- clustered_customers_k5.csv")
print("- pca_scatter_k5.png")
print("- cluster_summary_k5.json")
print("- cluster_summary_table_k5.csv (Resumo de Negócios para K=5)")

print("\nScript completed successfully.")
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style='whitegrid')

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'raw' / 'online_retail.csv'
MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.dropna(subset=['CustomerID'])
    work = work[~work['InvoiceNo'].astype(str).str.startswith('C')]
    work = work[(work['Quantity'] > 0) & (work['UnitPrice'] > 0)]
    work['InvoiceDate'] = pd.to_datetime(work['InvoiceDate'])
    work['TotalAmount'] = work['Quantity'] * work['UnitPrice']

    snapshot_date = work['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = work.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalAmount', 'sum'),
    ).reset_index()

    # Light outlier capping for stability.
    for col in ['Recency', 'Frequency', 'Monetary']:
        q1 = rfm[col].quantile(0.25)
        q3 = rfm[col].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        rfm[col] = rfm[col].clip(lower=lower, upper=upper)

    return rfm


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')

    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    rfm = build_rfm(df)

    X_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Elbow diagnostics
    inertias = []
    k_values = list(range(2, 11))
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    sns.lineplot(x=k_values, y=inertias, marker='o')
    plt.title('Elbow Curve for KMeans')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'elbow_curve.png', dpi=140)
    plt.close()

    # Final model
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    rfm['Cluster'] = clusters

    sil = silhouette_score(X_scaled, clusters)

    # Segment distribution
    plt.figure(figsize=(7, 4))
    sns.countplot(data=rfm, x='Cluster', palette='Set2')
    plt.title('Customer Count by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Customer Count')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_distribution.png', dpi=140)
    plt.close()

    # Cluster profile heatmap
    profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    profile_norm = (profile - profile.mean()) / profile.std(ddof=0)

    plt.figure(figsize=(7, 4.5))
    sns.heatmap(profile_norm, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
    plt.title('Normalized Cluster Profile (RFM)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_profile_heatmap.png', dpi=140)
    plt.close()

    # 2D scatter for visual intuition
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=np.log1p(rfm['Recency']),
        y=np.log1p(rfm['Monetary']),
        hue=rfm['Cluster'].astype(str),
        palette='tab10',
        alpha=0.75,
        s=35,
    )
    plt.title('Cluster Scatter (log Recency vs log Monetary)')
    plt.xlabel('log(1 + Recency)')
    plt.ylabel('log(1 + Monetary)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_scatter_recency_monetary.png', dpi=140)
    plt.close()

    # Save model artifacts expected by app.
    joblib.dump(scaler, MODELS_DIR / 'rfm_scaler.joblib')
    joblib.dump(kmeans, MODELS_DIR / 'kmeans_rfm_model.joblib')

    # Save summary files.
    summary = {
        'n_customers': int(len(rfm)),
        'n_clusters': 4,
        'silhouette_score': float(sil),
        'cluster_counts': rfm['Cluster'].value_counts().sort_index().to_dict(),
        'cluster_profile_mean': profile.round(2).to_dict(),
    }
    with open(REPORTS_DIR / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    lines = [
        '# Results Summary',
        '',
        f"- Customers modeled: **{summary['n_customers']}**",
        f"- Clusters: **{summary['n_clusters']}**",
        f"- Silhouette score: **{summary['silhouette_score']:.4f}**",
        '',
        '## Cluster Counts',
    ]
    for k, v in summary['cluster_counts'].items():
        lines.append(f"- Cluster {k}: {v}")

    lines += [
        '',
        '## Exported Figures',
        '- reports/figures/elbow_curve.png',
        '- reports/figures/cluster_distribution.png',
        '- reports/figures/cluster_profile_heatmap.png',
        '- reports/figures/cluster_scatter_recency_monetary.png',
    ]

    (REPORTS_DIR / 'RESULTS_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Assets generated successfully.')


if __name__ == '__main__':
    main()

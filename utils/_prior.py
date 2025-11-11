import scanpy as sc
import numpy as np
import pandas as pd
import pyranges as pr
from anndata import AnnData
from typing import Optional, Callable
import os
import ot
from scipy.spatial.distance import cdist
from scipy.io import mmread
from scipy.sparse import coo_matrix
import scipy.sparse


def create_feat_prior(
    adata_rna: AnnData,
    adata_atac: AnnData,
    window_size: int = 1e5,
    return_dist = False, 
) -> pd.DataFrame:

    hvg_df = adata_rna.var[adata_rna.var['highly_variable']].copy()
    hvg_gene_names = hvg_df.index.values

    hvg_df['WindowStart'] = hvg_df['chromStart'] - window_size
    hvg_df['WindowEnd'] = hvg_df['chromEnd'] + window_size
    hvg_df['WindowStart'] = hvg_df['WindowStart'].clip(lower=0)

    windows_df = hvg_df[['chrom', 'WindowStart', 'WindowEnd', 'gene_name']].rename(
        columns={'chrom': 'Chromosome', 'WindowStart': 'Start', 'WindowEnd': 'End'}
    )
    gene_windows_pr = pr.PyRanges(windows_df)
    
    try:
        peaks_df = adata_atac.var_names.to_series().str.split('[:-]', expand=True)
        peaks_df.columns = ['Chromosome', 'Start', 'End']
        peaks_df['Start'] = pd.to_numeric(peaks_df['Start'])
        peaks_df['End'] = pd.to_numeric(peaks_df['End'])
        peaks_df['Name'] = adata_atac.var_names
        peaks_pr = pr.PyRanges(peaks_df)
    except Exception as e:
        raise ValueError(f"Failed to parse ATAC peaks. Ensure format is 'chr:start-end'. Error: {e}")

    overlapping_join = gene_windows_pr.join(peaks_pr)
    
    if overlapping_join.empty:
        print("No peaks found within any gene windows. Returning an empty DataFrame.")
        return pd.DataFrame(index=hvg_gene_names)
    print(f"Found {len(overlapping_join)} total gene-peak links.")
    overlap_df = overlapping_join.df
    linkage_matrix = pd.crosstab(
        overlapping_join.df['gene_name'],
        overlapping_join.df['Name']
    )
    final_matrix = linkage_matrix.reindex(hvg_gene_names, fill_value=0)
    final_matrix = (final_matrix > 0).astype(int)

    sorted_gname = []
    sorted_cname = []
    for gname in adata_rna.var.index:
        if gname in hvg_gene_names:
            sorted_gname.append(gname)
    for cname in adata_atac.var.index:
        if cname in final_matrix.columns:
            sorted_cname.append(cname)
    final_matrix = final_matrix.loc[sorted_gname, sorted_cname]

    print(f"\n✅ Function complete. Matrix shape: {final_matrix.shape}")

    if return_dist:
        gene_pos = hvg_df.set_index('gene_name')[['chromStart', 'chromEnd']]
    
        # pull arrays
        gene_names_arr = overlap_df['gene_name'].values
        peak_names_arr = overlap_df['Name'].values
    
        # get original gene starts/ends aligned to overlap rows
        gene_starts = gene_pos.loc[gene_names_arr, 'chromStart'].to_numpy()
        gene_ends   = gene_pos.loc[gene_names_arr, 'chromEnd'].to_numpy()
    
        # peak intervals from join (these are the real peak coords)
        p_start = overlap_df['Start_b'].to_numpy()
        p_end   = overlap_df['End_b'].to_numpy()
    
        # distance between peak interval and ORIGINAL gene body [gene_starts, gene_ends]
        # if they overlap: 0
        distance = np.where(
            p_end < gene_starts,             # peak completely before gene
            gene_starts - p_end,
            np.where(
                gene_ends < p_start,         # peak completely after gene
                p_start - gene_ends,
                0
            )
        )
    
        dist_df = pd.DataFrame({
            'gene_name': gene_names_arr,
            'peak': peak_names_arr,
            'distance': distance.astype(int)
        })
    
        # dedupe: one (gene, peak) → smallest distance
        dist_df = (
            dist_df
            .groupby(['gene_name', 'peak'], as_index=False)['distance']
            .min()
        )
    
        # 7) keep only pairs that actually survived in final_matrix
        fm_nonzero = final_matrix.stack()
        fm_nonzero = fm_nonzero[fm_nonzero != 0].reset_index()
        fm_nonzero.columns = ['gene_name', 'peak', 'value']
    
        dist_nonzero_df = fm_nonzero.merge(
            dist_df,
            on=['gene_name', 'peak'],
            how='left'
        ).drop(columns=['value'])
    
        dist_nonzero_df['distance'] = dist_nonzero_df['distance'].fillna(0).astype(int)
        return final_matrix, dist_nonzero_df
    return final_matrix

def create_samp_prior(rna, atac, imbalance:bool=False, reg = 0.05, reg_m = 1):
    rna = rna.copy()
    geneactivity_atac = AnnData(X=atac.obsm['geneactivity_scores'], obs=pd.DataFrame(index=atac.obs.index.values), var=pd.DataFrame(index=atac.uns['geneactivity_names']))
    common_genes = list(set(rna.var_names) & set(geneactivity_atac.var_names))
    rna = rna[:, common_genes].copy()
    gam = geneactivity_atac[:, common_genes].copy()
    
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.normalize_total(gam, target_sum=1e4)
    sc.pp.log1p(gam)
    
    cm_hvg_genes = sc.pp.highly_variable_genes(rna, n_top_genes=2000, subset=False, inplace=False)
    rna = rna[:, cm_hvg_genes["highly_variable"]].copy()
    gam = gam[:, cm_hvg_genes["highly_variable"]].copy()
    
    sc.pp.scale(rna)
    sc.pp.scale(gam)
    dist = cdist(rna.X, gam.X, metric="correlation")
    p = ot.unif(dist.shape[0])
    q = ot.unif(dist.shape[1])
    if imbalance:
        pi_samp = ot.unbalanced.sinkhorn_knopp_unbalanced(p, q, dist, reg=reg, reg_m=reg_m)
    else:
        pi_samp = ot.sinkhorn(p, q, dist, reg=reg)    
    return pi_samp

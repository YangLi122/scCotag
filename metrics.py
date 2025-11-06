import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from scipy.sparse.csgraph import connected_components
import pandas as pd
import anndata as ad
import scanpy as sc


def eva_foscttm_ot(transport_plan, cell_count = None, partial=False):
    import torch
    if transport_plan.shape[0] == transport_plan.shape[1]:
        cell_count = transport_plan.shape[0]
    # Check for NaN or Inf (compatible with numpy and torch)
    if isinstance(transport_plan, torch.Tensor):
        if torch.isinf(transport_plan).any() or torch.isnan(transport_plan).any():
            print("Error: Transportation plan contains NaN or Inf.")
            return False
        diag = torch.diag(transport_plan)
        foscttm_x = (transport_plan >= diag.unsqueeze(1)).float().mean(dim=1)
        foscttm_y = (transport_plan >= diag.unsqueeze(0)).float().mean(dim=0)
        foscttm = (foscttm_x.sum() + foscttm_y.sum()) / (2 * cell_count)
        return foscttm.item()

    elif isinstance(transport_plan, np.ndarray):
        if np.isinf(transport_plan).any() or np.isnan(transport_plan).any():
            print("Error: Transportation plan contains NaN or Inf.")
            return False
        diag = np.diag(transport_plan)
        foscttm_x = (transport_plan >= np.expand_dims(diag, axis=1)).mean(axis=1)
        foscttm_y = (transport_plan >= np.expand_dims(diag, axis=0)).mean(axis=0)
        foscttm = (foscttm_x.sum() + foscttm_y.sum()) / (2 * cell_count)
        return float(foscttm)


def eva_foscttm_emb(embeddings, cell_count = None):
    import torch
    if embeddings.shape[0] == embeddings.shape[1]:
        cell_count = embeddings.shape[0]
    if isinstance(embeddings, torch.Tensor):
        if torch.isinf(embeddings).any() or torch.isnan(embeddings).any():
            print("Error: Embedding matrix contains NaN or Inf.")
            return False
        diag = torch.diag(embeddings)
        foscttm_x = (embeddings < diag.unsqueeze(1)).float().mean(dim=1)
        foscttm_y = (embeddings < diag.unsqueeze(0)).float().mean(dim=0)
        foscttm = (foscttm_x.sum() + foscttm_y.sum()) / (2 * cell_count)
        return foscttm.item()

    elif isinstance(embeddings, np.ndarray):
        if np.isinf(embeddings).any() or np.isnan(embeddings).any():
            print("Error: Embedding matrix contains NaN or Inf.")
            return False
        diag = np.diag(embeddings)
        foscttm_x = (embeddings < np.expand_dims(diag, axis=1)).mean(axis=1)
        foscttm_y = (embeddings < np.expand_dims(diag, axis=0)).mean(axis=0)
        foscttm = (foscttm_x.sum() + foscttm_y.sum()) / (2 * cell_count)
        return float(foscttm)

    elif isinstance(embeddings, np.ndarray):
        if np.isinf(embeddings).any() or np.isnan(embeddings).any():
            print("Error: Embedding matrix contains NaN or Inf.")
            return False
        diag = np.diag(embeddings)
        foscttm_x = (embeddings < np.expand_dims(diag, axis=1)).mean(axis=1)
        foscttm_y = (embeddings < np.expand_dims(diag, axis=0)).mean(axis=0)
        foscttm = (foscttm_x.sum() + foscttm_y.sum()) / (2 * cell_count)
        return float(foscttm)


def for_test_only(inference_output):
    gene_cell_emb = inference_output['rna_inference_res']['z'].detach().cpu().numpy()
    peak_cell_emb = inference_output['atac_inference_res']['z'].detach().cpu().numpy()
    
    d = distance_matrix(gene_cell_emb, peak_cell_emb)
    cell_count = gene_cell_emb.shape[0]
    return eva_foscttm_emb(d, cell_count)

def foscttm(x: np.ndarray, y: np.ndarray,
            **kwargs):
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    foscttm_mean = np.mean([foscttm_x.mean(), foscttm_y.mean()])
    return foscttm_mean

def nearest_cell_celltype(meta, mod_1, mod_2):
    if mod_1.shape != mod_2.shape:
        raise ValueError("Shapes do not match!")
    d = distance_matrix(mod_1, mod_2)
    dist = pd.DataFrame(d)
    dist.index = meta
    dist.columns = meta

    min_col_in_row = dist.idxmin(axis = 1)
    nearest_mod2_of_rna = min_col_in_row.index == min_col_in_row.values
    nearest_mod2_of_rna_ratio = sum(nearest_mod2_of_rna)/len(nearest_mod2_of_rna)

    min_row_in_col = dist.idxmin()
    nearest_rna_of_mod2 = min_row_in_col.index == min_row_in_col.values
    nearest_rna_of_mod2_ratio = sum(nearest_rna_of_mod2)/len(nearest_rna_of_mod2)

    nearest_cell_cell_type_mean = (nearest_mod2_of_rna_ratio+nearest_rna_of_mod2_ratio)/2
    
    nearest_cell_cell_type_list = pd.Series([nearest_mod2_of_rna_ratio, nearest_rna_of_mod2_ratio, nearest_cell_cell_type_mean], 
                                            index=['RNA', 'ATAC', 'Mean'])
    return nearest_cell_cell_type_list


def adjusted_rand_index(rna, atac, rna_cell_type, atac_cell_type):
    rna_ari = adjusted_rand_score(rna.obs[rna_cell_type], rna.obs['louvain'])
    atac_ari = adjusted_rand_score(atac.obs[atac_cell_type], atac.obs['louvain'])
    mean_ari = (rna_ari + atac_ari) / 2 
    return mean_ari


def adjusted_mutual_info_index(rna, atac, rna_cell_type, atac_cell_type):
    rna_ami = adjusted_mutual_info_score(rna.obs[rna_cell_type], rna.obs['louvain'], average_method='arithmetic')
    atac_ami = adjusted_mutual_info_score(atac.obs[atac_cell_type], atac.obs['louvain'], average_method='arithmetic')
    mean_ami = (rna_ami + atac_ami) / 2 
    return mean_ami


def graph_connectivity(rna, atac, rna_cell_type, atac_cell_type, domain, n_neibor=15):
    def graph_connectivity_manual(adata, label_key="cell_type"):
        C = adata.obsp["connectivities"].tocsr()
        labels = adata.obs[label_key].astype("category")
        cats = [c for c in labels.cat.categories if (labels == c).sum() > 0]
    
        per_label = {}
        for c in cats:
            idx = np.where(labels.values == c)[0]
            if idx.size <= 1:
                per_label[c] = 1.0  # convention
                continue
            C_sub = C[idx[:, None], idx]
            n_comp, comp_labels = connected_components(C_sub, directed=False, connection='weak')
            largest = np.bincount(comp_labels).max()
            per_label[c] = largest / idx.size
        return float(np.mean(list(per_label.values())))

    obs_rna  = rna.obs.copy()
    obs_atac = atac.obs.copy()
    label_key = "cell_type" 
    if label_key not in obs_rna.columns:
        obs_rna[label_key] = obs_rna[rna_cell_type]
    if label_key not in obs_atac.columns:
        obs_atac[label_key] = obs_atac[atac_cell_type]
    
    obs = pd.concat([obs_rna[[label_key, domain]],
                 obs_atac[[label_key, domain]]],
                axis=0)
    X = np.vstack([rna.obsm['latent'], atac.obsm['latent']]).astype(np.float32) 
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["latent"] = X
    sc.pp.neighbors(adata, n_neighbors=n_neibor, use_rep='latent', metric="euclidean")
    overall = graph_connectivity_manual(adata, label_key=label_key)
    return overall



def silhouette_index(rna, atac, rna_cell_type, atac_cell_type, domain, n_neibor=15):
    import scib
    obs_rna  = rna.obs.copy()
    obs_atac = atac.obs.copy()
    label_key = "cell_type" 
    if label_key not in obs_rna.columns:
        obs_rna[label_key] = obs_rna[rna_cell_type]
    if label_key not in obs_atac.columns:
        obs_atac[label_key] = obs_atac[atac_cell_type]
    
    obs = pd.concat([obs_rna[[label_key, domain]],
                 obs_atac[[label_key, domain]]],
                axis=0)
    X = np.vstack([rna.obsm['latent'], atac.obsm['latent']]).astype(np.float32) 
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["latent"] = X
    sc.pp.neighbors(adata, n_neighbors=n_neibor, use_rep='latent', metric="euclidean")
    asw = scib.metrics.silhouette_batch(adata, batch_key=domain, group_key=label_key, 
                        embed='latent', metric="euclidean", return_all=False, scale=True, verbose=False)
    return asw


def score_summary(df):
    invert = ["FOSCTTM"]  # these will be flipped so that higher is better
    df_norm = df.copy()
    for col in df.columns:
        vals = df[col]
        if col in invert:
            vals = 1 - (vals - vals.min()) / (vals.max() - vals.min())
        else:
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        df_norm[col] = vals
    
    # --- 3. Compute the summary as mean of normalized metrics ---
    df_norm["Summary"] = df_norm.mean(axis=1)
    
    # --- 4. Sort descending by Summary ---
    df_norm = df_norm.sort_values("Summary", ascending=False)
    return df_norm



def nearest_cell_celltype_topk(meta, mod_1, mod_2, topk=1):
    mod_1 = np.asarray(mod_1)
    mod_2 = np.asarray(mod_2)
    if mod_1.shape != mod_2.shape:
        raise ValueError("Shapes do not match!")
    n = mod_1.shape[0]
    if len(meta) != n:
        raise ValueError("Length of meta must equal number of rows in embeddings.")
    if topk < 1:
        raise ValueError("topk must be >= 1.")
    k = min(topk, n)

    # pairwise distances and label-indexed DataFrame
    d = distance_matrix(mod_1, mod_2)  # shape (n, n)
    dist = pd.DataFrame(d, index=pd.Index(meta, name="mod1_label"),
                           columns=pd.Index(meta, name="mod2_label"))

    # mod_1 -> mod_2: for each row, take top-k smallest columns and
    # compute the fraction of those whose label equals the row label
    def frac_match_row(s: pd.Series) -> float:
        topk_idx = s.nsmallest(k).index  # labels of the k nearest mod_2 cells
        return np.mean(topk_idx == s.name)

    # mod_2 -> mod_1: for each column, do the analogous computation
    def frac_match_col(s: pd.Series) -> float:
        topk_idx = s.nsmallest(k).index  # labels of the k nearest mod_1 cells
        return np.mean(topk_idx == s.name)

    rna_topk_frac = dist.apply(frac_match_row, axis=1).mean()
    atac_topk_frac = dist.T.apply(frac_match_col, axis=1).mean()

    mean_score = (rna_topk_frac + atac_topk_frac) / 2.0
    scores = pd.Series(
        [rna_topk_frac, atac_topk_frac, mean_score],
        index=['RNA', 'ATAC', 'Mean'],
        name=f'topk={k}'
    )
    return scores

def nearest_cell_celltype_topk_imbalance(meta_1, meta_2, mod_1, mod_2, topk=1):
    mod_1 = np.asarray(mod_1)
    mod_2 = np.asarray(mod_2)
    if mod_1.shape != mod_2.shape:
        raise ValueError("Shapes do not match!")
    n1 = mod_1.shape[0]
    n2 = mod_2.shape[0]
    if len(meta_1) != n1 or len(meta_2) != n2:
        raise ValueError("Length of meta must equal number of rows in embeddings.")
    if topk < 1:
        raise ValueError("topk must be >= 1.")
    k = min(topk, n1)

    # pairwise distances and label-indexed DataFrame
    d = distance_matrix(mod_1, mod_2)  # shape (n, n)
    dist = pd.DataFrame(d, index=pd.Index(meta_1, name="mod1_label"),
                           columns=pd.Index(meta_2, name="mod2_label"))

    # mod_1 -> mod_2: for each row, take top-k smallest columns and
    # compute the fraction of those whose label equals the row label
    def frac_match_row(s: pd.Series) -> float:
        topk_idx = s.nsmallest(k).index  # labels of the k nearest mod_2 cells
        return np.mean(topk_idx == s.name)

    # mod_2 -> mod_1: for each column, do the analogous computation
    def frac_match_col(s: pd.Series) -> float:
        topk_idx = s.nsmallest(k).index  # labels of the k nearest mod_1 cells
        return np.mean(topk_idx == s.name)

    rna_topk_frac = dist.apply(frac_match_row, axis=1).mean()
    atac_topk_frac = dist.T.apply(frac_match_col, axis=1).mean()

    mean_score = (rna_topk_frac + atac_topk_frac) / 2.0
    scores = pd.Series(
        [rna_topk_frac, atac_topk_frac, mean_score],
        index=['RNA', 'ATAC', 'Mean'],
        name=f'topk={k}'
    )
    return scores



# def within_modality_distortion(imbalance_atac_emb, paired_atac_emb, 
#                                    unique_cell_index, share_cell_index):
#     dist_gt = cdist(paired_atac_emb[unique_cell_index], paired_atac_emb[share_cell_index])
#     dist = cdist(imbalance_atac_emb[unique_cell_index], imbalance_atac_emb[share_cell_index])
#     r = stats.pearsonr(dist.ravel(), dist_gt.ravel()).statistic
#     distortation = 1 - r
#     return distortation



# def cross_modality_distortion(imbalance_rna_emb, paired_rna_emb, imbalance_atac_emb, paired_atac_emb, rna_cell_index, atac_cell_index):
#     dist_gt = cdist(paired_atac_emb[atac_cell_index], paired_rna_emb[rna_cell_index])
#     dist = cdist(imbalance_atac_emb[atac_cell_index], imbalance_rna_emb[rna_cell_index])
#     r = stats.pearsonr(dist.ravel(), dist_gt.ravel()).statistic
#     distortation = 1 - r
#     return distortation

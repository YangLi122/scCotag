import scanpy as sc
import pandas as pd
import numpy as np
import torch
from scipy.sparse import coo_matrix
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import ot
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from scipy.stats import ks_2samp
from scipy import stats
import math
import pathlib 
import random
import sys
import warnings
import argparse
import re
PARENT = pathlib.Path(__file__).resolve().parents[1]   # one level up
sys.path.insert(0, str(PARENT))
from utils._data import lsi
from utils.model.scCotag import scCotag
from utils.metrics import eva_foscttm_emb, eva_foscttm_ot
from utils._prior import create_feat_prior, create_samp_prior
from utils._data import alignability_estimation, fit_gmm_and_classify
from utils.model.ot import coot


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-rna", dest="input_rna", type=pathlib.Path, required=True,
        help="Path to input RNA dataset (.h5ad)"
    )
    parser.add_argument(
        "--input-atac", dest="input_atac", type=pathlib.Path, required=True,
        help="Path to input ATAC dataset (.h5ad)"
    )
    parser.add_argument(
        "--seeds", dest="seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="List of random seeds to run (e.g., --seeds 1 2 3 4 5)"
    )
    parser.add_argument(
        "--train-dir", dest="train_dir", type=pathlib.Path, required=True,
        help="Base directory where training logs and checkpoints are stored"
    )
    parser.add_argument(
        "--output-rna", dest="output_rna", type=pathlib.Path, required=True,
        help="Path of output RNA latent file (.csv) (seed suffix will be added)"
    )
    parser.add_argument(
        "--output-atac", dest="output_atac", type=pathlib.Path, required=True,
        help="Path of output ATAC latent file (.csv) (seed suffix will be added)"
    )
    parser.add_argument(
        "--imbalance", dest="imbalance", default=False,
        help="Path of output ATAC latent file (.csv) (seed suffix will be added)"
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    """Best-effort RNG control across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_seed_suffix(path: pathlib.Path, seed: int) -> pathlib.Path:
    """Insert .seed{N} before suffix, e.g. results.csv -> results.seed3.csv"""
    return path.with_name(f"{path.stem}.seed{seed}{path.suffix}")

def add_best_before_seed(path: pathlib.Path) -> pathlib.Path:
    """
    Insert '.best' immediately before the first '.seed{N}' in the filename.
    Examples:
      results.seed3.csv   -> results.best.seed3.csv
    """
    # Try to put ".best" right before the first ".seed{N}"
    new_name, n = re.subn(r'\.seed(\d+)', r'.best.seed\1', path.name, count=1)
    if n > 0:
        return path.with_name(new_name)

def run_sccotag(args: argparse.Namespace, seed: int) -> None:
    print(f"\n===== Seed {seed} =====")
    set_all_seeds(seed)
    
    train_dir = args.train_dir / f"seed-{seed}"
    out_rna = add_seed_suffix(args.output_rna, seed)
    out_atac = add_seed_suffix(args.output_atac, seed)

    print("[1/4] Reading data...")
    rna = ad.read_h5ad(args.input_rna)
    atac = ad.read_h5ad(args.input_atac)
    rna.layers["counts"] = rna.X.copy()
    atac.layers["counts"] = atac.X.copy()

    print("[2/4] Preprocessing...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    lsi(atac, n_components=100, n_iter=15)
    
    rna = rna[:,~rna.var.chrom.isna()]
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
    rna.var['gene_name'] = rna.var.index
    rna.X = rna.layers["counts"].copy()
    atac.X = atac.layers["counts"].copy()
    
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.filter_genes(atac, min_counts=20)
    samp_prior = create_samp_prior(rna, atac, args.imbalance)
    feat_prior = create_feat_prior(rna, atac, 5e3)
    if args.imbalance:
        alignability = alignability_estimation(rna, atac)
        alignability = alignability.pow(2)
        thre = np.max(fit_gmm_and_classify(alignability)['means'])

    feat_prior = feat_prior.loc[feat_prior.sum(axis=1) != 0, feat_prior.sum(axis=0) != 0]
    gene_names = list(feat_prior.index.values)
    peak_names = list(feat_prior.columns.values)
    rna = rna[:, gene_names]
    atac = atac[:, peak_names]
    rna.layers["counts"] = rna.X.copy()
    atac.layers["counts"] = atac.X.copy()
    
    sparse_matrix = coo_matrix(feat_prior.values)
    gene_indices = torch.from_numpy(sparse_matrix.row).to(torch.long)
    peak_indices = torch.from_numpy(sparse_matrix.col).to(torch.long)
        
    offset_peak_indices = peak_indices + len(feat_prior)
    edge_index = torch.stack([gene_indices, offset_peak_indices], dim=0)
    edge_index = to_undirected(edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=feat_prior.values.shape[0] + feat_prior.values.shape[1])[0]
        
    data = Data()
    data['num_genes'] = feat_prior.values.shape[0]
    data['num_peaks'] = feat_prior.values.shape[1]
    data['num_nodes'] = feat_prior.values.shape[0] + feat_prior.values.shape[1]
    data['edge_index'] = edge_index
    
    
    print("[3/4] Training scCotag...")
    if not args.imbalance:
        model = scCotag(rna=rna, atac=atac, graph=data, samp_prior=samp_prior,  feat_prior=feat_prior.values, 
                 rna_in_layer='X_pca', atac_in_layer='X_lsi', samp_alignment_weight=1e-1, vae_kl_weight=0.1, 
                distribution_alignment_weight=1e-2, graph_reweighting=True, device='cuda:0')

    else:
        model = scCotag(rna=rna, atac=atac, graph=data, samp_prior=samp_prior,  feat_prior=feat_prior.values, 
                 rna_in_layer='X_pca', atac_in_layer='X_lsi', samp_alignment_weight=1e-1, vae_kl_weight=0.1, 
                distribution_alignment_weight=1e-2, graph_reweighting=True, device='cuda:0', imbalance=args.imbalance, 
                       alignability=alignability, confidence_thre=thre)
    
    model.model_train(max_epochs = 4000, lr = 1e-3, weight_decay = 1e-4, warmup_epochs=1000) 


    print("[4/4] Saving results...")
    rna.obsm['scCotag'] = model.encoding_data(modality='rna', use_best = False)
    atac.obsm['scCotag'] = model.encoding_data(modality='atac', use_best = False)
    out_rna_df=pd.DataFrame(rna.obsm['scCotag'],index=rna.obs.index.values)
    out_atac_df=pd.DataFrame(atac.obsm['scCotag'],index=atac.obs.index.values)
    out_rna_df.to_csv(out_rna, header=False)
    out_atac_df.to_csv(out_atac, header=False)

    # out_rna = add_best_before_seed(out_rna)
    # out_atac = add_best_before_seed(out_atac)
    # rna.obsm['scCotag'] = model.encoding_data(modality='rna')
    # atac.obsm['scCotag'] = model.encoding_data(modality='atac')
    # out_rna_df=pd.DataFrame(rna.obsm['scCotag'],index=rna.obs.index.values)
    # out_atac_df=pd.DataFrame(atac.obsm['scCotag'],index=atac.obs.index.values)
    # out_rna_df.to_csv(out_rna, header=False)
    # out_atac_df.to_csv(out_atac, header=False)
    


def main(args: argparse.Namespace) -> None:
    seeds = args.seeds if args.seeds else [args.random_seed]
    for seed in seeds:
        run_sccotag(args, seed)


if __name__ == "__main__":
    main(parse_args())
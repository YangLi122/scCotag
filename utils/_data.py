import anndata as ad
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import issparse, coo_matrix
import torch
from anndata import AnnData
from typing import Optional, Union, Tuple, List
from sklearn.preprocessing import normalize
import sklearn
import re
import scipy
from math import ceil
from torch_geometric.data import Data
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture
Array = Union[np.ndarray, scipy.sparse.spmatrix]
RandomState = Optional[Union[np.random.RandomState, int]]
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

class DictDataset(Dataset):
    """
    Dataset class for dictionary data. It is a subclass of torch.utils.data.Dataset. It
    is meant to be used to load data from a dictionary where all entries are indexed by
    the same observations. The keys of the dictionary are the fields and the values are
    the data to be loaded.

    :param dict_data: dictionary with the data.
    """

    def __init__(self, dict_data):
        self.dict_data = dict_data

    def __getitem__(self, index: np.ndarray) -> dict:
        """
        Get the data for a given index.

        :param index: index of the observations to be retrieved.
        :return: a dictionary with the data for the given index.
        """
        out = {}
        for key, value in self.dict_data.items():
            if key == "cell_index":
                out[key] = list(np.array(value)[index])
            else:
                if issparse(value):
                    out[key] = torch.tensor(value[index].todense()).float()
                else:
                    out[key] = value[index]
        return out

    def split_train_val(self, ratio_val: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset into a training and a validation set.

        :param ratio_val: proportion of cells to be used for validation.
        :return: the indices of the observations to be used for training and validation.
        """
        n = len(self)
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = int(n * ratio_val)
        idx_val = idx[:n_val]
        idx_train = idx[n_val:]
        return idx_train, idx_val

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: the number of observations in the dataset.
        """
        key = "batch_index"
        return len(self.dict_data[key])


def dataset_configuration(adata: ad.AnnData,
                          in_key: str = None,
                          out_key: str = None,
                          batch_key: str = None,
                          device: str = 'cpu'):
    dic_data = {"cell_index": list(adata.obs_names)}
    if in_key == None:
        dic_data['input'] = torch.from_numpy(adata.layers["counts"].todense()).to(device)
    else:
        dic_data['input'] = torch.from_numpy(adata.obsm[in_key]).to(device)

    if out_key == None:
        dic_data['output'] = torch.from_numpy(adata.layers["counts"].todense()).to(device)
    else:
        dic_data['output'] = torch.from_numpy(adata.obsm[out_key]).to(device)

    # if batch_key == None:
    #     dic_data['batch_index'] = torch.zeros((adata.n_obs, 1), dtype=torch.long).to(device)
    # elif batch_key != None:
    #     dic_data['batch_index'] = torch.from_numpy(adata.obs[batch_key]).to(device)

    return (DictDataset(dic_data), dic_data['input'].shape[1], dic_data['output'].shape[1])

def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf




def lsi(
    adata: AnnData,
    n_components: int = 20,
    use_highly_variable: Optional[bool] = None,
    **kwargs,
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi







def vertex_degrees(
    eidx: np.ndarray,
    ewt: np.ndarray,
    vnum: Optional[int] = None,
    direction: str = "both",
) -> np.ndarray:
    r"""
    Compute vertex degrees

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    vnum
        Total number of vertices (determined by max edge index if not specified)
    direction
        Direction of vertex degree, should be one of {"in", "out", "both"}

    Returns
    -------
    degrees
        Vertex degrees
    """
    vnum = vnum or eidx.max() + 1
    adj = coo_matrix((ewt, (eidx[0], eidx[1])), shape=(vnum, vnum))
    if direction == "in":
        return adj.sum(axis=0).A1
    elif direction == "out":
        return adj.sum(axis=1).A1
    elif direction == "both":
        return adj.sum(axis=0).A1 + adj.sum(axis=1).A1 - adj.diagonal()
    raise ValueError("Unrecognized direction!")


def normalize_edges(
    eidx: np.ndarray, ewt: np.ndarray, method: str = "keepvar"
) -> np.ndarray:
    r"""
    Normalize graph edge weights

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    method
        Normalization method, should be one of {"in", "out", "sym", "keepvar"}

    Returns
    -------
    enorm
        Normalized weight of edges (:math:`n_{edges}`)
    """
    if method not in ("in", "out", "sym", "keepvar"):
        raise ValueError("Unrecognized method!")
    enorm = ewt
    if method in ("in", "keepvar", "sym"):
        in_degrees = vertex_degrees(eidx, ewt, direction="in")
        in_normalizer = np.power(in_degrees[eidx[1]], -1 if method == "in" else -0.5)
        in_normalizer[
            ~np.isfinite(in_normalizer)
        ] = 0  # In case there are unconnected vertices
        enorm = enorm * in_normalizer
    if method in ("out", "sym"):
        out_degrees = vertex_degrees(eidx, ewt, direction="out")
        out_normalizer = np.power(out_degrees[eidx[0]], -1 if method == "out" else -0.5)
        out_normalizer[
            ~np.isfinite(out_normalizer)
        ] = 0  # In case there are unconnected vertices
        enorm = enorm * out_normalizer
    return enorm

def get_default_numpy_dtype(complex: bool = False) -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    m = re.match(r"([^\.]+)\.([A-Za-z]+)(\d+)", str(torch.get_default_dtype()))
    _, dtype, bits = m.groups()
    if complex:
        dtype = "complex"
        bits = int(bits) * 2
    return getattr(np, f"{dtype}{bits}")

def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random

# @logged
class GraphDataset(Dataset):
    r"""
    Dataset for graphs with support for negative sampling

    Parameters
    ----------
    graph
        PyG Data object (must have edge_index; optional edge_weight, edge_sign)
    neg_samples
        Number of negative samples per edge
    weighted_sampling
        Whether to do negative sampling based on vertex importance
    deemphasize_loops
        Whether to deemphasize self-loops when computing vertex importance
    getitem_size
        Unitary fetch size for each __getitem__ call

    Note
    ----
    Custom shuffling performs negative sampling.
    """

    def __init__(
        self,
        graph: Data,                  
        neg_samples: int = 10,
        weighted_sampling: bool = True,
        deemphasize_loops: bool = True,
        getitem_size: int = 1,
    ) -> None:
        super().__init__()  
        self.eidx, self.ewt = self.graph2triplet(graph)  # CHANGED signature
        self.eset = {(i, j) for (i, j) in self.eidx.T}
        
        self.vnum = self.eidx.max() + 1
        if weighted_sampling:
            if deemphasize_loops:
                non_loop = self.eidx[0] != self.eidx[1]
                eidx = self.eidx[:, non_loop]
                ewt = self.ewt[non_loop]
            else:
                eidx = self.eidx
                ewt = self.ewt
            degree = vertex_degrees(eidx, ewt, vnum=self.vnum, direction="both")
        else:
            degree = np.ones(self.vnum, dtype=self.ewt.dtype)
        degree_sum = degree.sum()
        if degree_sum:
            self.vprob = degree / degree_sum  # Vertex sampling probability
        else:  # Possible when `deemphasize_loops` is set on a loop-only graph
            self.vprob = np.ones(self.vnum, dtype=self.ewt.dtype) / self.vnum

        effective_enum = self.ewt.sum()
        self.eprob = self.ewt / effective_enum  # Edge sampling probability
        self.effective_enum = round(effective_enum)

        self.neg_samples = neg_samples
        self.size = self.effective_enum * (1 + self.neg_samples)
        self.samp_eidx: Optional[np.ndarray] = None
        self.samp_ewt: Optional[np.ndarray] = None
        self.samp_esgn: Optional[np.ndarray] = None

    def graph2triplet(
        self,
        graph: Data,                    # CHANGED: was (graph: nx.Graph, vertices: pd.Index)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Convert PyG Data object to graph triplet

        Parameters
        ----------
        graph
            PyG Data object with fields:
              - edge_index: (2, E) long tensor
              - optional edge_weight: (E,) in (0,1]
              - optional edge_sign: (E,) in {-1, +1}

        Returns
        -------
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        ewt
            Weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)
        """
        default_dtype = get_default_numpy_dtype()

        # indices (assumed already integer-coded 0..N-1 in PyG)
        if not hasattr(graph, "edge_index") or graph.edge_index is None:
            raise ValueError("PyG Data must contain edge_index.")
        eidx = graph.edge_index.detach().cpu().numpy().astype(np.int64)
        if eidx.ndim != 2 or eidx.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")

        E = eidx.shape[1]

        # weights
        if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
            w = graph.edge_weight.detach().cpu().numpy()
        else:
            w = np.ones(E, dtype=default_dtype)
        ewt = np.asarray(w).astype(default_dtype)
        if ewt.shape[0] != E:
            raise ValueError("edge_weight length must equal number of edges.")
        if ewt.min() <= 0 or ewt.max() > 1:  # keep the reference's strict check
            raise ValueError("Invalid edge weight!")

        return eidx, ewt

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        s = slice(
            index * self.getitem_size, min((index + 1) * self.getitem_size, self.size)
        )
        return [
            torch.as_tensor(self.samp_eidx[:, s]),
            torch.as_tensor(self.samp_ewt[s]),
        ]

    def propose_shuffle(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (pi, pj), pw = self.eidx, self.ewt
        rs = get_rs(seed)

        # Sample positives by edge probability (with replacement)
        psamp = rs.choice(pw.size, self.effective_enum, replace=True, p=self.eprob)
        pi_, pj_, pw_ = pi[psamp], pj[psamp], pw[psamp]
        pw_ = np.ones_like(pw_)   # positive targets = 1

        # Build negatives: replicate sources; zero targets
        ni_ = np.tile(pi_, self.neg_samples)
        nw_ = np.zeros(pw_.size * self.neg_samples, dtype=pw_.dtype)
        nj_ = rs.choice(self.vnum, pj_.size * self.neg_samples, replace=True, p=self.vprob)

        # Avoid collisions with existing edges (i,j)
        # NOTE: Potential infinite loop if graph too dense (same as reference)
        remain = np.where([((int(i), int(j)) in self.eset) for i, j in zip(ni_, nj_)])[0]
        while remain.size:
            newnj = rs.choice(self.vnum, remain.size, replace=True, p=self.vprob)
            nj_[remain] = newnj
            remain = remain[
                [((int(i), int(j)) in self.eset) for i, j in zip(ni_[remain], newnj)]
            ]

        # Concatenate pos+neg, randomize order
        idx = np.stack([np.concatenate([pi_, ni_]), np.concatenate([pj_, nj_])])
        w   = np.concatenate([pw_, nw_])
        perm = rs.permutation(idx.shape[1])
        return idx[:, perm], w[perm]

    def accept_shuffle(
        self, shuffled: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        self.samp_eidx, self.samp_ewt = shuffled


@torch.no_grad()
def _topk_idx_per_row(M: torch.Tensor, k: Optional[int]) -> Optional[torch.Tensor]:
    if k is None or k >= M.shape[1]:
        return None
    return M.topk(k=k, dim=1).indices


@torch.no_grad()
def alignability_estimation(rna, atac):
    rna_count = rna.copy()
    common_genes = list(set(atac.uns['geneactivity_names']).intersection(set(rna_count.var.index.values)))
    geneactivity_atac = AnnData(X=atac.uns['geneactivity_scores'], obs=pd.DataFrame(index=atac.obs.index.values),    var=pd.DataFrame(index=atac.uns['geneactivity_names']))
    rna_common = rna_count[:, common_genes].copy()
    atac_common = geneactivity_atac[:, common_genes].copy()
    sc.pp.highly_variable_genes(rna_common, n_top_genes=2000, flavor="seurat_v3")
    hvg_genes_list = list(rna_common[:,rna_common.var['highly_variable'] == True].var.index.values)
    sc.pp.normalize_total(rna_common, target_sum=1e4)
    sc.pp.log1p(rna_common)
    sc.pp.normalize_total(atac_common, target_sum=1e4)
    sc.pp.log1p(atac_common)
    sc.pp.scale(rna_common)
    sc.pp.scale(atac_common)
    
    hvg_rna = rna_common[:, hvg_genes_list]
    hvg_atac = atac_common[:, hvg_genes_list]
    dist = torch.from_numpy(cdist(hvg_rna.X, hvg_atac.X, metric='cosine'))
    dist_rna = torch.from_numpy(cdist(hvg_rna.X, hvg_rna.X, metric='cosine'))
    dist_atac = torch.from_numpy(cdist(hvg_atac.X, hvg_atac.X, metric='cosine'))
    within_score = within_modality_score(dist_rna, dist_atac)
    cross_score = cross_modality_score(dist)
    scores = (within_score + cross_score) / 2
    med = torch.median(scores)
    mad = torch.median(torch.abs(scores - med)).clamp_min(1e-8)
    robust_scale = 1.4826 * mad
    
    z = (scores - med) / (robust_scale + 1e-10)       
    conf_score = torch.sigmoid(-z / 1)
    return conf_score
    


@torch.no_grad()
def within_modality_score(
    D_rna: torch.Tensor,   
    D_atac: torch.Tensor,  
    n_bins: int = 20,      
    tau_cost: float = 0.1, 
    tau_prob: float = 1.0, 
    eps: float = 1e-12
):
    """
    Returns:
      ent_atac:  (n_atac,) structural entropy in [0,1] (higher = ambiguous/unshared)
    """
    device = D_rna.device
    n_rna  = D_rna.shape[0]
    n_atac = D_atac.shape[0]

    Dr_sort = torch.sort(D_rna, dim=1).values[:, 1:]  
    Da_sort = torch.sort(D_atac, dim=1).values[:, 1:] 
    q = torch.linspace(0.0, 1.0, steps=n_bins+2, device=device)[1:-1]  
    idx_r = (q * (Dr_sort.shape[1]-1)).round().long()                 
    idx_a = (q * (Da_sort.shape[1]-1)).round().long()

    R_sig = Dr_sort[:, idx_r]   
    A_sig = Da_sort[:, idx_a]   

    def zrow(X):
        mu = X.mean(dim=1, keepdim=True)
        sd = X.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (X - mu) / sd
    R_sig = zrow(R_sig)
    A_sig = zrow(A_sig)
    C = torch.cdist(A_sig, R_sig, p=1) 
    logits = -C / (tau_prob + eps)              
    P = torch.softmax(logits, dim=1)               
    ent = -(P * (P + eps).log()).sum(dim=1) / torch.log(torch.tensor(P.shape[1], device=device).float())
    ent_atac = ent.clamp(0, 1)

    return ent_atac

@torch.no_grad()
def cross_modality_score(
    D_cross: torch.Tensor,  
    k: int = 10,      
):
    """
    Returns:
      a_k:    (n_atac,) raw aggregate (lower is better)
    """
    n_rna, n_atac = D_cross.shape
    kc = min(k, n_rna)
    vals, _ = torch.topk(D_cross, kc, dim=0, largest=False)  
    a_k = vals.mean(dim=0)
    return a_k


def fit_gmm_and_classify(values, *, higher_is_alignable=False, 
                         prior_weight=None, prob_cut=0.5, random_state=42):
    """
    values: 1D array of your statistic (e.g., mean of k min cross distances per ATAC cell)
    higher_is_alignable: False if lower values mean more alignable (distances); True for scores
    prior_weight: optional tuple (w_alignable, w_unalignable) to bias the fit; None = learned
    prob_cut: posterior cutoff to call a cell "alignable" (0.5 default; try 0.8 for higher precision)
    """
    x = np.asarray(values).reshape(-1, 1)

    # Fit a 2-component 1D GMM
    gmm = GaussianMixture(
        n_components=2, covariance_type="full",
        random_state=random_state, init_params="kmeans", reg_covar=1e-6
    )
    if prior_weight is not None:
        w_a, w_u = prior_weight
        gmm.weights_init = np.array([w_a, w_u]) / (w_a + w_u)
    gmm.fit(x)

    # Identify which component is "alignable"
    means = gmm.means_.ravel()         # shape (2,)
    covs  = gmm.covariances_.ravel()   # shape (2,)
    weights = gmm.weights_.ravel()
    if higher_is_alignable:
        idx_align = np.argmax(means)
    else:
        idx_align = np.argmin(means)   # lower mean -> alignable for distances

    # Posterior probability of alignable component
    resp = gmm.predict_proba(x)[:, idx_align]  # P(alignable | x_i)
    pred = resp >= prob_cut

    # Bayes decision boundary where posteriors are equal (optional; for plotting)
    # Solve: w1 N(x|m1,v1) = w2 N(x|m2,v2)
    def bayes_thresholds(w1, m1, v1, w2, m2, v2):
        # quadratic ax^2 + bx + c = 0 in x
        a = 0.5*(1/v2 - 1/v1)
        b = m1/v1 - m2/v2
        c = 0.5*((m2**2)/v2 - (m1**2)/v1) + np.log((w2*np.sqrt(v2)) / (w1*np.sqrt(v1)))
        # Handle near-equal variances
        if abs(a) < 1e-12:
            return np.array([ -c / b ])
        disc = b*b - 4*a*c
        if disc < 0:  # numerical guard
            return np.array([])
        roots = np.array([(-b - np.sqrt(disc))/(2*a), (-b + np.sqrt(disc))/(2*a)])
        return np.sort(roots)

    idx_other = 1 - idx_align
    thr = bayes_thresholds(weights[idx_align], means[idx_align], covs[idx_align],
                           weights[idx_other], means[idx_other], covs[idx_other])

    return {
        "post_align": resp,            # posterior P(alignable | x)
        "pred_align": pred,            # boolean labels by prob_cut
        "gmm": gmm,                    # fitted model (means, covs, weights)
        "align_component": idx_align,  # 0 or 1
        "bayes_thresholds": thr,       # typically one value between means
        "means": means, "vars": covs, "weights": weights
    }


def get_rs(random_state=None):
    """Simple random-state helper compatible with ints/None/RandomState."""
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(None if random_state is None else int(random_state))


def regulatory_inference(
    features: pd.Index,
    feature_embeddings: Union[np.ndarray, List[np.ndarray]],
    pyg_graph,  # torch_geometric.data.Data (expects `edge_index`)
    alternative: str = "two.sided",
    random_state=None,
):
    r"""
    Regulatory inference based on feature embeddings (PyTorch Geometric version)

    Parameters
    ----------
    features
        Feature names as a pandas Index, length must equal number of graph nodes
    feature_embeddings
        A single ndarray (n_features, d) or a list of ndarrays from 1+ models,
        each shaped (n_features, d_i). Rows must correspond to graph node IDs.
    pyg_graph
        PyG graph (expects `.edge_index` with shape [2, E])
    alternative
        One of {"two.sided", "less", "greater"}
    random_state
        Optional seed / RandomState

    Returns
    -------
    pyg_graph
        Same PyG graph with `edge_attr` appended as a (E, 3) tensor:
        columns are [score, pval, qval] for each edge in `edge_index` order.
        Also adds `edge_attr_names = ['score','pval','qval']` for convenience.
    """
    # ---------- Input normalization & checks ----------
    if isinstance(feature_embeddings, np.ndarray):
        feature_embeddings = [feature_embeddings]

    n_features_from_emb = {arr.shape[0] for arr in feature_embeddings}
    if len(n_features_from_emb) != 1:
        raise ValueError("All feature embeddings must have the same number of rows!")
    n_features = n_features_from_emb.pop()

    # Graph node count must match features & embeddings
    if not hasattr(pyg_graph, "num_nodes"):
        # try to infer from edge_index if num_nodes is missing
        num_nodes = int(pyg_graph.edge_index.max().item()) + 1
    else:
        num_nodes = int(pyg_graph.num_nodes)

    if n_features != features.shape[0]:
        raise ValueError("Feature embeddings do not match the number of feature names!")
    if n_features != num_nodes:
        raise ValueError(
            f"Graph node count ({num_nodes}) does not match features/embeddings ({n_features})."
        )

    # Ensure numpy arrays
    feature_embeddings = [np.asarray(arr, dtype=float) for arr in feature_embeddings]

    # ---------- Prepare normalized embeddings and permutations ----------
    rs = get_rs(random_state)

    # v: (n_features, n_models, d) after stacking; each model can have its own d,
    # so we first check/align that dimensions match across models by padding or reject.
    # The original code implicitly required same dim across models (np.stack).
    dims = {arr.shape[1] for arr in feature_embeddings}
    if len(dims) != 1:
        raise ValueError(
            "All feature embeddings must have the same number of columns (dimensions) to stack."
        )

    v = np.stack(feature_embeddings, axis=1)  # (n_features, n_models, d)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)  # L2-normalize last dim

    # Permuted background for each model (permute rows independently per model)
    vperm_models = []
    for arr in feature_embeddings:
        parr = rs.permutation(arr)  # permute features (rows)
        parr = parr / np.linalg.norm(parr, axis=-1, keepdims=True)
        vperm_models.append(parr)
    vperm = np.stack(vperm_models, axis=1)  # (n_features, n_models, d)

    # ---------- Edges ----------
    if not hasattr(pyg_graph, "edge_index"):
        raise ValueError("pyg_graph must have an 'edge_index' attribute.")

    edge_index = pyg_graph.edge_index
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    elif not isinstance(edge_index, np.ndarray):
        edge_index = np.asarray(edge_index)

    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    src = edge_index[0]
    dst = edge_index[1]
    E = src.shape[0]

    # ---------- Compute foreground (fg) and background (bg) ----------
    fg = np.empty(E, dtype=float)
    bg_list = []

    # Iterate in edge order to keep alignment with edge_index
    for i in tqdm(range(E), total=E, desc="regulatory_inference (PyG)"):
        s = int(src[i])
        t = int(dst[i])
        # dot per model -> (n_models,)
        dots_fg = (v[s] * v[t]).sum(axis=1)
        fg[i] = dots_fg.mean()

        # background from permuted rows (n_models,)
        dots_bg = (vperm[s] * vperm[t]).sum(axis=1)
        bg_list.append(dots_bg)

    # P-values from pooled background, identical to original approach
    bg = np.sort(np.concatenate(bg_list))  # shape: (n_models * E,)
    # Searchsorted operates elementwise if 'fg' is an array
    quantile = np.searchsorted(bg, fg, side="left") / bg.size

    if alternative == "two.sided":
        pvals = 2 * np.minimum(quantile, 1 - quantile)
    elif alternative == "greater":
        pvals = 1 - quantile
    elif alternative == "less":
        pvals = quantile
    else:
        raise ValueError('Unrecognized `alternative`! Use "two.sided", "less", or "greater".')

    # FDR correction
    qvals = fdrcorrection(pvals)[1]

    # ---------- Attach to PyG graph ----------
    edge_attr_np = np.stack([fg, pvals, qvals], axis=1)  # (E, 3)
    edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32)

    # If existing edge_attr present, either concatenate or overwrite.
    # To mimic the NX behavior (fresh attributes), we overwrite by default.
    pyg_graph.edge_attr = edge_attr
    pyg_graph.edge_attr_names = ["score", "pval", "qval"]  # helpful metadata

    # Optional: if you want to preserve human-readable sources/targets
    # pyg_graph.edge_sources = [features[i] for i in src]
    # pyg_graph.edge_targets = [features[i] for i in dst]

    return pyg_graph

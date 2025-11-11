import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import TransformerConv, GCNConv, InnerProductDecoder, GATv2Conv
import math
import torch.nn.functional as F
from torch_geometric.utils import softmax, negative_sampling, add_self_loops
from torch.distributions import Normal, kl_divergence, Bernoulli
from typing import Union, Tuple, Optional, Literal
from torch_geometric.data import Data
from ._nn_base import reparameterization_std
from .._data import normalize_edges, GraphDataset
EPS = 1e-10


class GraphConv(torch.nn.Module):
    ## propagation-only.
    ## No learnable parameters.
    ## Adopted from GLUE

    def forward(
        self,
        x: torch.Tensor,
        eidx: torch.Tensor,
        enorm: torch.Tensor
    ) -> torch.Tensor:
        sidx, tidx = eidx  # source index and target index

        # Message is the source node's features scaled by the normalization factor.
        # The sign multiplication is removed.
        message = x[sidx] * enorm.unsqueeze(1)

        res = torch.zeros_like(x)
        tidx = tidx.unsqueeze(1).expand_as(message)
        res.scatter_add_(0, tidx, message)  # Aggregate messages at the target nodes
        return res


class GATv2ConvReducerLayer(GATv2Conv):

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 dropout: float = 0.0,
                 scale_param: Optional[float] = 2.0,
                 global_norm: Literal["per_head", "all_heads"] = "per_head",
                 # pass-through kwargs: add_self_loops, edge_dim, bias, etc.
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         heads=heads,
                         dropout=dropout,
                         **kwargs)
        self.scale_param = scale_param
        self.global_norm = global_norm
        self.edge_scores: Optional[Tensor] = None  # [E, H]
        self.EPS = 1e-6

    def edge_update(self,
                    x_j: torch.Tensor, x_i: torch.Tensor,
                    edge_attr: Optional[torch.Tensor],
                    index: torch.Tensor,     
                    ptr: Optional[torch.Tensor],
                    dim_size: Optional[int]) -> torch.Tensor:
        # --- pre-activation (same as upstream GATv2) ---
        x = x_i + x_j  # [E, H, C]
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        logits = (x * self.att).sum(dim=-1) 

        if self.scale_param is not None:
            if self.global_norm == "per_head":
                mu = logits.mean(dim=0, keepdim=True)             
                sigma = logits.std(dim=0, keepdim=True).clamp_min(self.EPS)
            else:
                mu = logits.mean()
                sigma = logits.std().clamp_min(self.EPS)

            normed = (logits - mu) / (sigma / self.scale_param)
            alpha = torch.sigmoid(normed)                            
        else:
            alpha = softmax(logits, index, ptr, dim_size)           

        self.edge_scores = alpha
        self._alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha  # used by message(): out = x_j * alpha.unsqueeze(-1)


@torch.no_grad()
def prune_edges_by_threshold(edge_index: torch.Tensor,
                             alpha: torch.Tensor,
                             tau: float,
                             reduce: Literal["mean", "max"] = "mean") -> Tuple[torch.Tensor, torch.Tensor]:

    if alpha.dim() == 2:
        score = alpha.mean(dim=1) if reduce == "mean" else alpha.max(dim=1).values
    else:
        score = alpha
    keep = score >= tau
    return edge_index[:, keep], keep


@torch.no_grad()
def prune_edges_by_keep_ratio(edge_index: torch.Tensor,
                              alpha: torch.Tensor,
                              keep_ratio: float = 0.5,
                              reduce: Literal["mean", "max"] = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha.dim() == 2:
        score = alpha.mean(dim=1) if reduce == "mean" else alpha.max(dim=1).values
    else:
        score = alpha
    E = score.numel()
    k = max(1, int(math.ceil(E * keep_ratio)))
    # topk keeps largest scores
    topk = torch.topk(score, k=k, sorted=False)
    thresh = topk.values.min()
    keep = score >= thresh
    return edge_index[:, keep], keep


class GraphEncoder(nn.Module):
    def __init__(
        self, 
        num_nodes: int,
        out_channels: int, 
        scale: int = 3, 
    ):
        super().__init__()
        self.scale = scale
        self.emb = torch.nn.Parameter(torch.zeros(num_nodes, out_channels))
        self.conv = GraphConv()
        self.loc = nn.Linear(out_channels, out_channels)
        self.std_lin = nn.Linear(out_channels, out_channels)
        
    def forward(self, edge_index: torch.Tensor, edge_norm: torch.Tensor):
        ptr = self.conv(self.emb, edge_index, edge_norm)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        # var = torch.exp(torch.clamp(self.std_lin(ptr), min=-10, max=10)) + EPS
        # dist = Normal(loc, var.sqrt())
        # emb = reparameterization(loc, var) if self.training else loc
        dist = Normal(loc, std)
        emb = reparameterization_std(loc, std) if self.training else loc
        return dist, emb
        
class GraphDecoder(nn.Module):
    def forward(self, z: torch.Tensor, eidx: torch.Tensor) -> Bernoulli:
        sidx, tidx = eidx
        # z = F.normalize(z, p=2, dim=1)
        logits = (z[sidx] * z[tidx]).sum(dim=1)
        return Bernoulli(logits=logits)
        

class GraphAutoEncoder(nn.Module):
    def __init__(
        self, 
        graph: Data, 
        num_nodes: int, 
        out_channels: int, 
        scale: int = 3, 
        reconstruction_weight: int = 1, 
        divergence_weight: float = 0, 
        num_neg_edges: int = 10, 
        device: str = 'cpu',
        
    ):
        super().__init__()
        self.data = graph
        self.num_nodes = num_nodes
        self.scale = scale
        self.reconstruction_weight = reconstruction_weight
        self.divergence_weight = divergence_weight
        self.num_neg_edges = num_neg_edges
        self.encoder = GraphEncoder(num_nodes, out_channels, self.scale)
        self.decoder = GraphDecoder()
        self.device = device
        self.mp_edge_index = graph.edge_index.to(self.device)
        mp_ei_np = self.mp_edge_index.detach().cpu().numpy()
        if 'ewt' in graph:
            mp_enorm_np = normalize_edges(mp_ei_np, graph['ewt'].detach().cpu().numpy())
        else:
            mp_enorm_np = normalize_edges(mp_ei_np, np.ones(mp_ei_np.shape[1], dtype=np.float32))
        self.mp_edge_norm = torch.from_numpy(mp_enorm_np).to(self.device).float()
        
        self.gds = GraphDataset(
            graph=graph,
            neg_samples=num_neg_edges,
            # weighted_sampling=weighted_sampling,
            # deemphasize_loops=deemphasize_loops,
            getitem_size=1  # not used if we call propose_shuffle directly
        )
        self._eidx: Optional[Tensor] = None
        self._ewt: Optional[Tensor] = None
        
        self.to(self.device)

    @torch.no_grad()
    def set_epoch(self, seed: int):

        eidx_np, ewt_np = self.gds.propose_shuffle(seed=seed)
        self._eidx = torch.as_tensor(eidx_np, device=self.device, dtype=torch.long)
        self._ewt  = torch.as_tensor(ewt_np,  device=self.device, dtype=torch.float32)

    def edge_weights_tuning(self, pi_feat, prior, gamma=1.5):
        pos_mask = (self._ewt > 0)
        eidx_pos = self._eidx[:, pos_mask]
        eidx_src, eidx_dst = eidx_pos
        
        G, P = pi_feat.shape
        pi_feat = (pi_feat.detach() * prior.detach()).to(dtype=torch.float32)
        # pi = np.clip(pi_feat * prior, 0.0, None) ** gamma
        pi = pi_feat.clamp_min(0).pow(gamma)
        divisor = (pi * prior).sum(dim=1) / (prior.sum(axis=1) + EPS)
        pi = (pi / divisor.unsqueeze(1)).clamp(min=0, max=1)
        
        src_is_gene   = eidx_src < G
        src_is_peak   = eidx_src >= G
        dst_is_gene = eidx_dst < G
        dst_is_peak = eidx_dst >= G
    
        mask_g2p = src_is_gene & dst_is_peak
        mask_p2g = src_is_peak & dst_is_gene
        lossw = torch.ones(len(eidx_src)).to(self.device)
        if mask_g2p.any():
            g = eidx_src[mask_g2p]               
            p = eidx_dst[mask_g2p] - G                 
            w_g2p = pi[g, p].to(lossw.dtype)
            lossw[mask_g2p] = w_g2p
    
        if mask_p2g.any():
            p = eidx_src[mask_p2g] - G              
            g = eidx_dst[mask_p2g]                     
            w_p2g = pi[g, p].to(lossw.dtype)
            lossw[mask_p2g] = w_p2g
        lossw = lossw.clamp_min(0).clamp_max(1)
        return lossw
    
    
    def inference(self, x):
        dist, emb = self.encoder(self.mp_edge_index, self.mp_edge_norm)
        # dist, emb = self.encoder(self.mp_edge_index, self.mp_edge_norm)
        return dict(dist=dist, z=emb)
        
    def reconstruction_loss(self, z, lossw=None):

        dist = self.decoder(z, self._eidx)
        per_edge_nll = -dist.log_prob(self._ewt)   # [E]
        if lossw is not None:
            pos_mask = (self._ewt > 0)
            neg_mask = ~pos_mask
            pos_nll = per_edge_nll[pos_mask]
            neg_nll = per_edge_nll[neg_mask]
            w_pos = lossw
            w_neg = torch.ones_like(neg_nll)
            pos_mean = ((w_pos * pos_nll).sum() / (w_pos.sum().clamp_min(EPS))
                        if pos_nll.numel() else per_edge_nll.new_tensor(0.0))
            neg_mean = ((w_neg * neg_nll).sum() / (w_neg.sum().clamp_min(EPS))
                        if neg_nll.numel() else per_edge_nll.new_tensor(0.0))     
            have_pos = 1 if pos_nll.numel() else 0
            have_neg = 1 if neg_nll.numel() else 0
            avgc = max(have_pos + have_neg, 1)
            g_nll = ( (neg_mean if have_neg else 0.0) + (pos_mean if have_pos else 0.0) ) / avgc
        else: 
            pos_mask = (self._ewt > 0).to(torch.int64) 
            n_pos = int(pos_mask.sum().item()) 
            n_neg = int(per_edge_nll.numel() - n_pos) 
            sums = torch.zeros(2, dtype=per_edge_nll.dtype, device=per_edge_nll.device) 
            sums.scatter_add_(0, pos_mask, per_edge_nll) 
            avgc = (1 if n_pos > 0 else 0) + (1 if n_neg > 0 else 0) 
            g_nll = (sums[0] / max(n_neg, 1) + sums[1] / max(n_pos, 1)) / max(avgc, 1)
        return g_nll

    def divergence_loss(self, inference, num_nodes):
        kl_loss = kl_divergence(inference['dist'], Normal(0, 1)).sum(dim=-1)
        return kl_loss.mean() / num_nodes
        
    def loss(self, inference, epoch, pi_feat, prior, reweighting=True):
        self.set_epoch(seed = epoch)
        
        if reweighting:
            lossw = self.edge_weights_tuning(pi_feat, prior)
        else: 
            lossw = None
        recon_loss = self.reconstruction_loss(inference['z'], lossw)
        kl_loss = self.divergence_loss(inference, self.data['num_nodes'])
        loss = self.reconstruction_weight * recon_loss + self.divergence_weight * kl_loss
        return {"loss":loss, "reconstruction_loss": self.reconstruction_weight * recon_loss, "kl_loss": self.divergence_weight * kl_loss}
        













        

import torch
from torch import nn
import numpy as np
import pandas as pd
import anndata as ad
import scanpy 
from torch_geometric.data import Data
from copy import deepcopy
from torch import optim
from torch.optim.lr_scheduler import StepLR
import ot
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import scipy.sparse as sp
from geomloss import SamplesLoss
import torch.nn.functional as F
from tqdm import tqdm
from typing import Literal

from ._nn_base import AutoEncoder
from ._graph_base import GraphAutoEncoder
from .ot import coot_emb, dist, coot, ucoot
from ..metrics import for_test_only, eva_foscttm_ot, eva_foscttm_emb
from .._data import _topk_idx_per_row, alignability_estimation

EPS = 1e-10

class scCotag(nn.Module):
    def __init__(
        self,
        rna: ad.AnnData, 
        atac: ad.AnnData, 
        graph: Data, 
        hidden_dims: list = [256, 256], 
        emb_dim: int = 50, 
        samp_prior = None, 
        feat_prior = None, 
        rna_in_layer = None,
        atac_in_layer = None,
        rna_out_layer = None,
        atac_out_layer = None,
        prior_weight: float = 0.5, 
        vae_kl_weight: float = 0.1, 
        rna_vae_weight: int = 1, 
        atac_vae_weight: int = 1,
        graph_vae_weight: float = 1, 
        samp_alignment_weight: float = 0.01, 
        distribution_alignment_weight: float = 0.01, 
        topk = None,
        graph_reweighting: bool = True,
        ot_reg: float = 0.05,
        ot_reg_m: float = 1, 
        imbalance: bool = False, 
        alignability = None, 
        regulartory_weight = 0.1, 
        confidence_thre = 0.8,
        device: str = "cpu",
    ):
        super().__init__()
        self.rna = rna
        self.atac = atac
        self.graph = graph
        self.device = device

        self.prior_weight = np.clip(prior_weight, a_min = 0, a_max=1)
        self.vae_kl_weight = vae_kl_weight
        self.rna_vae_weight = rna_vae_weight
        self.atac_vae_weight = atac_vae_weight
        self.graph_vae_weight = graph_vae_weight
        self.samp_alignment_weight = samp_alignment_weight
        self.distribution_alignment_weight = distribution_alignment_weight
        self.graph_reweighting = graph_reweighting
        self.topk = topk
        self.ot_reg = ot_reg
        self.ot_reg_m = ot_reg_m
        self.imbalance = imbalance
        self.regulartory_weight = regulartory_weight
        self.confidence_thre = confidence_thre
        self.alignability = alignability

        self.prior = dict()
        if samp_prior is not None:
            self.prior['samp_prior'] = torch.from_numpy(samp_prior).to(self.device)
        if feat_prior is not None:
            self.prior['feat_prior'] = torch.from_numpy(feat_prior).to(self.device)

        self.rnaAE = AutoEncoder(adata=rna, emb_dim=emb_dim, in_key=rna_in_layer, out_key=rna_out_layer, 
                                encoder_hiddens=hidden_dims, divergence_weight=vae_kl_weight, device=device)
        self.atacAE = AutoEncoder(adata=atac, emb_dim=emb_dim, in_key=atac_in_layer, out_key=atac_out_layer, 
                                 encoder_hiddens=hidden_dims, divergence_weight=vae_kl_weight, device=device)
        self.graphAE = GraphAutoEncoder(graph=graph, num_nodes=self.graph['num_nodes'], out_channels=emb_dim, 
                                        divergence_weight=vae_kl_weight,  device=device)

        self.pi_samp = None
        self.pi_feat = None
        ## Debugging purpose
        self.rnaEmb = None
        self.atacEmb = None
        self.geneEmb = None
        self.peakEmb = None
        self.epoch = None
        self.to(self.device)

    def _kl_beta(self, epoch: int, warmup_epochs: int | None) -> float:
        if not warmup_epochs or warmup_epochs <= 0:
            return self.vae_kl_weight
        t = max(0, min(epoch, warmup_epochs))
        kl_beta = self.vae_kl_weight * min(1.0, t / float(warmup_epochs))
        self.rnaAE.divergence_weight  = kl_beta
        self.atacAE.divergence_weight = kl_beta
        self.graphAE.divergence_weight = kl_beta

    def _align_beta(self, epoch: int, warmup_epochs: int | None) -> float:
        # t = max(0, min(epoch, warmup_epochs))
        t = max(0, (epoch - warmup_epochs))
        self.crt_samp_alignment_weight = self.samp_alignment_weight * min(1.0, t / float(warmup_epochs))
    
    def _distribution_beta(self, epoch: int, warmup_epochs: int | None) -> float:
        t = max(0, (epoch - warmup_epochs))
        self.crt_distribution_alignment_weight = self.distribution_alignment_weight * min(1.0, t / float(warmup_epochs))
    
    def run_ot(self):
        rna = self.rna.copy()
        atac = self.atac.copy()
        rna.X = rna.layers["counts"]
        atac.X = atac.layers["counts"]
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        X = rna.X.copy().todense()
        Y = atac.X.copy().todense()
        prior = dict()
        prior['prior_samp'] = self.prior['samp_prior'].detach().cpu().numpy()
        prior['prior_feat'] = self.prior['feat_prior'].detach().cpu().numpy()
        if not self.imbalance:
            self.pi_samp, self.pi_feat = coot(X, Y, prior, epsilon=self.ot_reg)
        else:
            self.pi_samp, self.pi_feat = ucoot(X, Y, prior, epsilon=self.ot_reg)
            
        self.pi_samp = torch.from_numpy(self.pi_samp).to(self.device).float()
        self.pi_feat = torch.from_numpy(self.pi_feat).to(self.device).float()
        
    
    def inference(self):
        rna_inference_res = self.rnaAE.inference(self.rnaAE.dataset.dict_data)
        atac_inference_res = self.atacAE.inference(self.atacAE.dataset.dict_data)
        feature_inference_res = self.graphAE.inference(self.graph)
        
        self.rnaEmb = rna_inference_res['z']
        self.atacEmb = atac_inference_res['z']
        self.geneEmb = feature_inference_res['z'][:self.graph['num_genes']]
        self.peakEmb = feature_inference_res['z'][self.graph['num_genes']:]
        return {"rna_inference_res": rna_inference_res, "atac_inference_res": atac_inference_res, 
                "feature_inference_res": feature_inference_res}

    def samp_alignment_loss(self, warmup_epochs, anchor: Literal["none", "rna", "atac"] = "rna", gamma=2):
        
        if anchor == 'rna':
            z_r, z_a = self.rnaEmb.detach(), self.atacEmb
        elif anchor == 'atac':
            z_r, z_a = self.rnaEmb, self.atacEmb.detach()
        else:
            z_r, z_a = self.rnaEmb, self.atacEmb
    
        if self.prior_weight == 1:
            P = self.pi_samp.detach()
        
        else:   
            p = torch.full((z_r.shape[0],), 1.0 / z_r.shape[0], device=self.device, dtype=self.rnaEmb.dtype)
            q =   torch.full((z_a.shape[0],), 1.0 / z_a.shape[0], device=self.device, dtype=self.atacEmb.dtype)  
            res = ot.solve_sample(
                z_r, z_a, a=p, b=q,
                reg=self.ot_reg,
                metric="sqeuclidean",
                method="geomloss_tensorized",
                unbalanced=None,
                unbalanced_type="KL",
                grad="autodiff",   
                verbose=False
                )
            
            if hasattr(res, "plan") and res.plan is not None:
                P = res.plan     
            elif hasattr(res, "lazy_plan") and res.lazy_plan is not None:
                P = res.lazy_plan[:]      
            P = P.detach()
            P0 = self.pi_samp
            P = (1 - self.prior_weight) * P + self.prior_weight * P0
            
        P = P.clamp_min(0).pow(gamma)
        total_loss = torch.zeros((), device=self.device)

        col_sum = P.sum(dim=0, keepdim=True)            
        P_col = P / col_sum                                              
        Ybar = P_col.t() @ z_r                                     
        if self.imbalance:
            def _make_anchor_mask(conf, thr=0.8):
                return conf >= thr

            def _anchor_distance_preservation(
                Z_before,        
                Z_target,          
                anchor_mask, 
                *, 
                scale_invariant=True,
                sammon=True,
                beta=0.7,
                eps=1e-12):
                device = Z_before.device
                idxA = torch.nonzero(anchor_mask, as_tuple=False).squeeze(1)
                if idxA.numel() == 0:
                    return Z_before.new_tensor(0.0)
            
                ZA = Z_before[idxA]                
                TA = Z_target[idxA]                 
            
                # For all non-anchors, compute distances to anchors
                idxN = torch.nonzero(~anchor_mask, as_tuple=False).squeeze(1)
                if idxN.numel() == 0:
                    return Z_before.new_tensor(0.0)
            
                ZN = Z_before[idxN]           
                TN = Z_target[idxN]                  
            
                Db = torch.cdist(ZN, ZA)           
                Da = torch.cdist(TN, TA)           
            
                if scale_invariant:
                    s = ((Db * Da).sum(dim=1) / (Da.pow(2).sum(dim=1) + eps)).unsqueeze(1)  # (nN,1)
                    Da_s = s * Da
                else:
                    Da_s = Da
            
                if sammon:
                    W = 1.0 / (Db + eps)
                    err = (W * (Da_s - Db)).pow(2).mean(dim=1) 
                    med = torch.median(torch.sqrt(err + eps))
                    loss_vec = torch.sqrt(err + eps) / (beta * med + eps)
                else:
                    rmse = torch.sqrt(((Da_s - Db).pow(2).mean(dim=1)) + eps)
                    scale = Db.mean(dim=1).clamp_min(eps)
                    loss_vec = rmse / (beta * scale + eps)
            
                return loss_vec.mean()

            atac_confidence = self.alignability.detach().clamp(0, 1)
            anchor_mask = _make_anchor_mask(atac_confidence, thr=self.confidence_thre)
            m = torch.where(anchor_mask, torch.ones_like(atac_confidence), atac_confidence)

            loss_anchor = _anchor_distance_preservation(
                    Z_before=z_a, Z_target=Ybar, anchor_mask=anchor_mask,
                    scale_invariant=True, sammon=True, beta=1
                )
            loss = torch.mean((z_a - Ybar) ** 2) * z_r.shape[1]
            loss = loss_anchor * self.regulartory_weight + loss
            return loss

        else:
            loss = torch.mean((z_a - Ybar) ** 2) * z_r.shape[1]
            return loss



        
    def distribution_alignment_loss(self):
        mixing_module = SamplesLoss(
            loss="sinkhorn", p=2, scaling=0.9, blur = 0.01, backend="tensorized")

        cost = mixing_module(self.rnaEmb, self.atacEmb)
        cost = torch.clamp_min(cost, 0.0) 
        return cost
    
    def compute_loss(self, inference_res, warmup_epochs):
        rna_generative_res = self.rnaAE.generative(inference_res['rna_inference_res'], self.geneEmb)
        atac_generative_res = self.atacAE.generative(inference_res['atac_inference_res'], self.peakEmb)

        rna_vae_loss = self.rnaAE.loss(self.rnaAE.dataset.dict_data, inference_res['rna_inference_res'], rna_generative_res)
        atac_vae_loss = self.atacAE.loss(self.atacAE.dataset.dict_data, inference_res['atac_inference_res'], atac_generative_res)
        graph_vae_loss = self.graphAE.loss(inference_res['feature_inference_res'], self.epoch, self.pi_feat, self.prior['feat_prior'], self.graph_reweighting)
        atac_scalar = self.rna.layers['counts'].mean() / self.atac.layers['counts'].mean()

        if self.epoch > warmup_epochs and self.samp_alignment_loss != 0:
            samp_align_loss = self.samp_alignment_loss(warmup_epochs)
        else:
            samp_align_loss = torch.zeros((), device=self.device)
        
        if self.epoch > warmup_epochs and self.distribution_alignment_weight != 0:
            distribution_align_loss = self.distribution_alignment_loss()
        else:
            distribution_align_loss = torch.zeros((), device=self.device)
        rna_vae_loss_wt = self.rna_vae_weight  * rna_vae_loss['loss']
        atac_vae_loss_wt = self.atac_vae_weight  * atac_vae_loss['loss'] * atac_scalar
        graph_vae_loss_wt = self.graph_vae_weight *  graph_vae_loss['loss']
        samp_align_loss_wt = self.crt_samp_alignment_weight * samp_align_loss
        distribution_align_loss_wt = self.crt_distribution_alignment_weight * distribution_align_loss

        loss = rna_vae_loss_wt + atac_vae_loss_wt + graph_vae_loss_wt + samp_align_loss_wt + distribution_align_loss_wt
        return {"loss": loss, "rna_vae_loss": rna_vae_loss_wt, "atac_vae_loss": atac_vae_loss_wt, "graph_vae_loss": graph_vae_loss_wt, "samp_align_loss": samp_align_loss_wt, "distribution_align_loss": distribution_align_loss_wt, "rna_kl_loss":  rna_vae_loss['kl_loss'], "rna_recon_loss": rna_vae_loss['reconstruction_loss'], "atac_kl_loss":  atac_vae_loss['kl_loss'], "atac_recon_loss": atac_vae_loss['reconstruction_loss'], "graph_kl_loss": graph_vae_loss['kl_loss'], "graph_recon_loss": graph_vae_loss['reconstruction_loss']}

        
    def model_train(self, max_epochs=500, lr=1e-3, weight_decay=1e-4, warmup_epochs=500):
        self.run_ot()
        self.best_model = None
        self.min_loss = np.inf
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        print("Training Starts...")
        self.train()
        for epoch in tqdm(range(max_epochs)):
            self.epoch = epoch
            self._kl_beta(epoch, warmup_epochs)
            self._align_beta(epoch, warmup_epochs)
            self._distribution_beta(epoch, warmup_epochs)
   
            optimizer.zero_grad()
            inference_out = self.inference()
            loss = self.compute_loss(inference_out, warmup_epochs)
            
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"loss={loss['loss']:.4f} | "
                    f"rna_vae_loss={loss['rna_vae_loss']:.4f} | "
                    f"rna_recon_loss={loss['rna_recon_loss']:.4f} | "
                    f"rna_kl_loss={loss['rna_kl_loss']:.4f} | "
                    f"atac_vae_loss={loss['atac_vae_loss']:.4f} | "
                    f"atac_recon_loss={loss['atac_recon_loss']:.4f} | "
                    f"atac_kl_loss={loss['atac_kl_loss']:.4f} | "
                    f"graph_vae_loss={loss['graph_vae_loss']:.4f} | "
                    f"graph_recon_loss={loss['graph_recon_loss']:.4f} | "
                    f"graph_kl_loss={loss['graph_kl_loss']:.4f} | "
                    f"samp_align_loss={loss['samp_align_loss']:.4f} | "
                    f"distribution_align_loss={loss['distribution_align_loss']:.4f}"
                )

            if loss['loss'] < self.min_loss and epoch > (warmup_epochs * 2):
                self.min_loss = loss['loss']
                self.best_model = deepcopy(self.state_dict())
            # torch.autograd.set_detect_anomaly(True)
            loss['loss'].backward()
            optimizer.step()

        print("Training Ends...")

    def encoding_data(self, modality: str = None, adata: ad.AnnData = None, use_best = True):
        if use_best:
            self.load_state_dict(self.best_model)
        self.eval()
        if modality == "rna":
            encoder = self.rnaAE
        elif modality == "atac":
            encoder = self.atacAE

        if adata is None:
            if modality == "rna":
                emb = encoder.inference(self.rnaAE.dataset.dict_data)['z']
            elif modality == "atac":
                emb = encoder.inference(self.atacAE.dataset.dict_data)['z']

        else:
            emb = encoder.inference(adata)['z']

        return emb.detach().cpu().numpy()

    def encoding_graph(self, graph):
        self.eval()
        emb = self.graphAE.inference(graph)['z']
        return emb.detach().cpu().numpy()
    




















from torch import nn
import torch.nn.functional as F
import torch
import collections
from collections.abc import Callable, Iterable
from typing import Literal
from torch.distributions import Normal, kl_divergence
import anndata as ad
import numpy as np
from torch.autograd import Function
import torch.distributions as D

from .._data import dataset_configuration
# from .prob import NB, ZINB, EPS
EPS = 1e-10

def reparameterization_var(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def reparameterization_std(mu, std):
    return Normal(mu, std).rsample()


class MLPLayers(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dims: list, 
        dropout: float = 0, 
        use_batch_norm: bool = False,
        last_activation: bool = False,
    ):
        super().__init__()
        layers = []
        for i in range(len(hidden_dims)):
            block = [
                nn.Linear(in_dim, hidden_dims[i]),
            ]
            if use_batch_norm:
                block.append(nn.BatchNorm1d(hidden_dims[i]))
            if i < len(hidden_dims) - 1 or last_activation:
                block.append(nn.LeakyReLU(negative_slope=0.2))
            block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))
            in_dim = hidden_dims[i]
            
        # layers.append(nn.Linear(hidden_dims[-1], out_dim))

        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        out = self.mlp(x)
        return out


class DataEncoder(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_dims, 
        dropout: float = 0.2, 
        use_batch_norm: bool = True, 
        last_activation: bool = True, 
        return_dist: bool = True, 
    ):
        super().__init__()
        self.return_dist = return_dist
        self.encoder = MLPLayers(
            in_dim = in_dim, 
            hidden_dims = hidden_dims, 
            dropout = dropout,
            last_activation = last_activation, 
            use_batch_norm=use_batch_norm, 
        )
        self.mean_encoder = nn.Linear(hidden_dims[-1], out_dim)
        self.std_encoder = nn.Linear(hidden_dims[-1], out_dim)
        # self.var_activation = torch.exp() 
    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_std = F.softplus(self.std_encoder(q)) + EPS
        # q_v = torch.exp(torch.clamp(self.log_var_encoder(q), min=-10, max=10)) + EPS
        # dist = Normal(q_m, q_v.sqrt())
        dist = Normal(q_m, q_std)
        if self.training:
            latent = reparameterization_std(q_m, q_std)
        else:
            latent = q_m
        if self.return_dist:
            return dist, latent
        else:
            return latent

class DataDecoder(nn.Module):
    def __init__(
        self, 
        out_dim: int, 
        n_batch: int = 1, 
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(n_batch, out_dim))
        self.log_theta = nn.Parameter(torch.zeros(n_batch, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_batch, out_dim))

    def forward(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        l: torch.Tensor,
        batch_index: int = 0,
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale[batch_index])
        logit_mu = scale * (z @ v.t()) + self.bias[batch_index]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[batch_index]
        logits = (mu + EPS).log() - log_theta
        # logits = -(theta.log() - (mu + EPS).log())         
        return D.NegativeBinomial(total_count=log_theta.exp(), logits=logits)
        # log_theta = self.log_theta[batch_index]
        # return D.NegativeBinomial(log_theta.exp(), logits=(mu + EPS).log() - log_theta)


        
class AutoEncoder(nn.Module):
    def __init__(
        self, 
        adata: ad.AnnData, 
        emb_dim: int = 50, 
        in_key: str = None,
        out_key: str = None,
        encoder_hiddens: list = None, 
        reconstruction_weight: int = 1,
        divergence_weight: float = 0,
        dropout: float = 0.2, 
        device: str = "cpu", 
    ):
        super().__init__()
        self.adata = adata
        self.emb_dim = emb_dim
        self.in_key = in_key
        self.out_key = out_key
        self.reconstruction_weight = reconstruction_weight
        self.divergence_weight = divergence_weight
        self.device = device
        self.TOTAL_COUNT = 1e4

        self.dataset, in_dim, out_dim = dataset_configuration(adata = self.adata, in_key = in_key, out_key = out_key, device=self.device)

        self.encoder = DataEncoder(
            in_dim = in_dim, 
            out_dim = emb_dim, 
            hidden_dims = encoder_hiddens, 
            dropout = dropout,
            last_activation = True,
            
        )
        self.decoder = DataDecoder(
            out_dim = out_dim
        )

        self.to(self.device)
    def normalize(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
    
    def inference(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        library = x['output'].sum(dim=1, keepdim=True)
        if self.in_key is None:
            normx = self.normalize(x["input"], library)
            dist, emb = self.encoder(normx)
        else:
            dist, emb = self.encoder(x["input"])
        return dict(z=emb, dist = dist, library = library)
        
    def generative(self, samp_res: dict[str, torch.Tensor], feat_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        nb_dist = self.decoder(z=samp_res['z'], v = feat_emb, l=samp_res['library'])
        return nb_dist

    def reconstruction_loss(self, x: dict[str, torch.Tensor], reconstructed) -> torch.Tensor:
        recloss = -reconstructed.log_prob(x["output"]).nanmean()
        return recloss
        
    def divergence_loss(self, x: dict[str, torch.Tensor], inference: dict[str, torch.Tensor]):
        dist = inference["dist"]
        kl_loss = kl_divergence(dist, Normal(0,1)).sum(dim=-1)
        return kl_loss.mean() / x['output'].shape[1]
        
    def loss(self, 
             x: dict[str, torch.Tensor], 
             inference: dict[str, torch.Tensor],
             reconstructed) -> torch.Tensor:
        recon_loss = self.reconstruction_loss(x, reconstructed)
        kl_loss = self.divergence_loss(x, inference)
        loss = self.reconstruction_weight * recon_loss + self.divergence_weight * kl_loss
        return {"loss":loss, "reconstruction_loss": self.reconstruction_weight * recon_loss, "kl_loss": self.divergence_weight * kl_loss}
        
        

        
    













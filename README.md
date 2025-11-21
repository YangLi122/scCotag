# scCotag
scCotag: Diagonal integration of single-cell multi-omics data via prior-informed Co-optimal transport and regularized barycentric mapping

Installation:
```
git clone https://github.com/YangLi122/scCotag.git
cd scCotag
micromamba create -n sccotag -f environment.yml python=3.12
```


For testing purpose, we provide the subsampled PBMC data (perfectly paired) which can be downloaded via following link:
```
https://drive.google.com/file/d/1TRErY-7m1bKF5eak8g1VE0VFnFwd5HuN/view?usp=sharing
https://drive.google.com/file/d/1MxpLn7E9Tw8dWgse08y2j1VpPetSuHm7/view?usp=sharing
```

To run the model (replacing the input and output data path if needed):
```
micromamba activate sccotag
python Run_scCotag.py --input-rna 10x-Multiome-Pbmc-Subsampled_100perCT-RNA.h5ad --input-atac 10x-Multiome-Pbmc-Subsampled_100perCT-ATAC.h5ad --output-rna rna_out.h5ad --output-atac atac_out.h5ad --train-dir ./
```

For evaluating purpose, you may use the output .h5ad files which contains cell embeddings and feature embeddings:
```
from utils.metrics import foscttm
import scanpy as sc
import anndata as ad

rna_out = sc.read_h5ad('./rna_out.h5ad')
atac_out = sc.read_h5ad('./atac_out.h5ad')

## e.g., to calculate the FOSCTTM score
foscttm(rna_out.obsm['scCotag'], atac_out.obsm['scCotag'])

## or, to vislalize
combined = ad.concat([rna_out, atac_out])
sc.pp.neighbors(combined, use_rep="scCotag", metric="cosine")
sc.tl.umap(combined)
sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65)
```

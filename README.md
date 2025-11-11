# scCotag

Installation:
```
micromamba create -n sccoatg -f environment.yml python=3.12
```


For testing purpose, we provide the subsampled PBMC data (perfectly paired) which can be downloaded via following link:
```
https://drive.google.com/file/d/1MxpLn7E9Tw8dWgse08y2j1VpPetSuHm7/view?usp=sharing
https://drive.google.com/file/d/1TRErY-7m1bKF5eak8g1VE0VFnFwd5HuN/view?usp=drive_link
```

To run the model:
```
micromamba activate sccoatg
python Run_scCotag.py --input-rna 10x-Multiome-Pbmc-Subsampled_100perCT-RNA.h5ad --input-atac 10x-Multiome-Pbmc-Subsampled_100perCT-ATAC.h5ad --output-rna rna_emb_out.csv --output-atac atac_emb_out.csv --train-dir ./

```

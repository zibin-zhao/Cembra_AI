# Toy Data for Pipeline Testing

Generated with `scripts/generate_toy_data.py`

## Contents
- `toy_gwas_eur.tsv`: Simulated EUR GWAS summary statistics (500 SNPs, N=400,000)
- `toy_gwas_eas.tsv`: Simulated EAS GWAS summary statistics (500 SNPs, N=50,000)
- `toy_ld_eur.npz`: EUR LD matrix (500 × 500, block-diagonal)
- `toy_ld_eas.npz`: EAS LD matrix
- `toy_genotype.npy`: Genotype dosage matrix (200 × 500)
- `toy_phenotype.tsv`: Sample info with binary phenotype (knee OA)
- `toy_bim.tsv`: SNP information (like PLINK .bim)
- `toy_annotations.tsv`: Functional annotation scores

## Parameters
- Seed: 42
- Causal SNPs: 50
- Prevalence: ~15%
- LD blocks: 20

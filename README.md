# VCC - CRISPR Perturbation Gene Expression Dataset

A comprehensive analysis and preprocessing pipeline for the VCC (CRISPR Perturbation) gene expression dataset.

## 📁 Project Structure

```
VCC/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── environment.yml                        # Conda environment configuration
│
├── data/                                  # Data directory
│   ├── raw/                               # Raw data files (downloaded)
│   │   ├── adata_Training.h5ad            # Training set (221K cells × 18K genes)
│   │   ├── gene_names.csv                 # Gene name mappings
│   │   └── pert_counts_Validation.csv     # Validation perturbation counts
│   └── processed/                         # Processed data (generated)
│
├── scripts/                               # Executable Python scripts
│   ├── preview_data.py                    # Preview all data files
│   └── visualize_h5ad.py                  # Generate visualizations & analysis
│
├── analysis/                              # Analysis outputs
│   └── visualizations/                    # Generated plots
│       ├── h5ad_visualization.png         # Main analysis plots
│       └── h5ad_expression_details.png    # Expression matrix details
│
├── docs/                                  # Documentation
│   ├── VCC_DATA_DESCRIPTION.md            # Detailed data description
│   └── h5ad_visualize.md                  # H5AD visualization guide
│
└── notebooks/                             # Jupyter notebooks (optional)
    └── (place analysis notebooks here)
```

## 🚀 Quick Start

### Setup Environment

```bash
# Create conda environment
conda create -n vcc python=3.11 -y
conda activate vcc

# Install dependencies
pip install anndata pandas numpy scanpy h5py matplotlib seaborn scipy
```

### Preview Data

```bash
cd /home/richard/projects/VCC
conda run -n vcc python scripts/preview_data.py
```

Output:
- 📊 Shape: 221,273 cells × 18,080 genes
- 🎯 151 unique target genes (150 perturbed + 1 control)
- 🔢 48 experimental batches
- 📊 50 validation target genes

### Generate Visualizations

```bash
conda run -n vcc python scripts/visualize_h5ad.py
```

Outputs saved to:
- `analysis/visualizations/h5ad_visualization.png` - 6-panel summary visualization
- `analysis/visualizations/h5ad_expression_details.png` - Expression matrix details

## 📊 Dataset Overview

### Training Data: `adata_Training.h5ad`

- **Cells (obs):** 221,273
- **Genes (vars):** 18,080
- **Size:** ~7.2 GB (sparse format)
- **Sparsity:** 51.69% zeros
- **Format:** AnnData H5AD

**Observation Metadata:**
- `target_gene` - CRISPR target gene (151 unique)
- `guide_id` - Guide RNA identifier (189 unique)
- `batch` - Experimental batch (48 unique)

**Gene Metadata:**
- `gene_id` - Ensembl gene identifier

**Expression Matrix:**
- Type: Sparse CSR matrix
- Data type: float32
- Non-zero elements: 1.93B

### Gene Names: `gene_names.csv`

- 18,079 gene names in order
- Corresponds to genes in expression matrix

### Validation Perturbations: `pert_counts_Validation.csv`

- 50 target genes for validation
- Cell counts: 161-2,925 cells per gene
- Median UMI per cell: ~54K

## 📈 Key Statistics

| Metric | Value |
|--------|-------|
| Total cells | 221,273 |
| Total genes | 18,080 |
| Control cells (non-targeting) | 38,176 |
| Perturbed cells | 183,097 |
| Unique target genes | 151 |
| Unique batches | 48 |
| Sparsity | 51.69% |
| Avg non-zero expression | 6.50 |

## 🔍 Documentation

- **[VCC_DATA_DESCRIPTION.md](docs/VCC_DATA_DESCRIPTION.md)** - Comprehensive data documentation with statistics and distributions
- **[h5ad_visualize.md](docs/h5ad_visualize.md)** - Guide to H5AD structure visualization

## 📦 Dependencies

Key libraries:
- `anndata` - AnnData format support
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scanpy` - Single-cell analysis
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scipy` - Scientific computing
- `h5py` - HDF5 file support

## 💡 Usage Examples

### Loading Data in Python

```python
import anndata as ad
import pandas as pd

# Load training data
adata = ad.read_h5ad('data/raw/adata_Training.h5ad')

# Access expression matrix
X = adata.X  # (221273, 18080) sparse matrix

# Access cell metadata
obs_df = adata.obs  # Cell annotations
print(obs_df['target_gene'].value_counts())

# Access gene metadata
var_df = adata.var  # Gene annotations
print(var_df['gene_id'])

# Load validation data
validation = pd.read_csv('data/raw/pert_counts_Validation.csv')
```

### Filtering Data

```python
# Select control cells
control_cells = adata[adata.obs['target_gene'] == 'non-targeting']

# Select specific batch
batch_cells = adata[adata.obs['batch'] == 'Flex_1_01']

# Select cells by target gene
target_perturbed = adata[adata.obs['target_gene'] == 'TMSB4X']
```

## 📝 Next Steps

1. **Explore data** - Run `preview_data.py` and `visualize_h5ad.py`
2. **Read documentation** - Check `docs/VCC_DATA_DESCRIPTION.md`
3. **Analyze patterns** - Look at batch effects, gene expression distributions
4. **Develop models** - Build prediction models using the expression data
5. **Create notebooks** - Add analysis notebooks to `notebooks/` directory

## 📄 License & Attribution

This repository contains analysis tools for the VCC dataset. 
Refer to the original dataset documentation for usage terms.

## 🤝 Contributing

To extend this project:
1. Add new scripts to `scripts/`
2. Update `docs/` with documentation
3. Save outputs to `analysis/`
4. Add analysis notebooks to `notebooks/`

---

**Last Updated:** November 12, 2025  
**Python Version:** 3.11  
**Conda Environment:** vcc


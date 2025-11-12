# VCC Project Manifest

## Directory Structure and Contents

### Root Level

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `requirements.txt` | Python package dependencies |
| `environment.yml` | Conda environment specification |
| `.gitignore` | Git ignore rules |
| `MANIFEST.md` | This file - project manifest |

### `/data` - Data Storage

#### `/data/raw` - Raw Data (Do Not Modify)
```
data/raw/
├── adata_Training.h5ad          (221K cells × 18K genes, ~7.2 GB)
├── gene_names.csv               (18,079 gene names)
└── pert_counts_Validation.csv   (50 validation targets)
```

**Key Info:**
- Training data format: AnnData H5AD
- Cells: 221,273 observations
- Genes: 18,080 variables
- Sparsity: 51.69% zeros
- Expression matrix: 1.93B non-zero values

#### `/data/processed` - Processed Data (To Be Generated)
```
data/processed/
├── (preprocessed datasets)
├── (filtered subsets)
└── (transformed data)
```

**Future use for:**
- Normalized expression matrices
- Filtered datasets
- Batch-corrected data
- Dimensionality reduction outputs

### `/scripts` - Python Scripts

```
scripts/
├── preview_data.py              Preview all data files
└── visualize_h5ad.py            Generate visualizations & statistics
```

#### `preview_data.py`
- **Purpose:** Quick overview of all 3 data files
- **Outputs:** Terminal display of data summaries
- **Usage:** `python scripts/preview_data.py`

#### `visualize_h5ad.py`
- **Purpose:** Detailed H5AD analysis and visualization
- **Outputs:** 
  - Tree structure of H5AD object
  - Statistical summaries
  - 6-panel visualization plot
  - Expression matrix details plot
- **Usage:** `python scripts/visualize_h5ad.py`

### `/analysis` - Analysis Outputs

#### `/analysis/visualizations` - Generated Plots

```
analysis/visualizations/
├── h5ad_visualization.png            6-panel summary plot
│   ├── Target gene distribution (top 20)
│   ├── Batch distribution (top 20)
│   ├── Gene expression sparsity
│   ├── Guide RNA type distribution
│   ├── Target × Batch heatmap
│   └── Mean expression per cell
│
└── h5ad_expression_details.png       Expression matrix details
    ├── Expression matrix sample (100×100)
    └── Non-zero value distribution
```

**Note:** All visualizations generated at 300 DPI

### `/docs` - Documentation

```
docs/
├── VCC_DATA_DESCRIPTION.md      Comprehensive data documentation
└── h5ad_visualize.md            H5AD visualization guide
```

#### `VCC_DATA_DESCRIPTION.md`
- **Sections:**
  - File overview
  - Training data dimensions & metadata
  - Target gene distribution
  - Batch information
  - Gene names structure
  - Validation set statistics
  - Key insights
  - Data format notes

#### `h5ad_visualize.md`
- **Sections:**
  - H5AD structure overview
  - Visualization methodology
  - Interpretation guide

### `/notebooks` - Jupyter Notebooks (Optional)

```
notebooks/
├── (exploratory analysis notebooks)
├── (model training notebooks)
└── (results visualization notebooks)
```

**To be created as needed for:**
- Data exploration
- Statistical analysis
- Model development
- Results visualization

## File Types & Data Sizes

### Data Files

| File | Format | Size | Records | Variables |
|------|--------|------|---------|-----------|
| adata_Training.h5ad | HDF5 (AnnData) | ~7.2 GB | 221,273 | 18,080 |
| gene_names.csv | CSV | ~200 KB | 18,079 | 1 |
| pert_counts_Validation.csv | CSV | ~2 KB | 50 | 3 |

### Script Files

| File | Language | Lines | Purpose |
|------|----------|-------|---------|
| preview_data.py | Python 3 | ~120 | Data preview |
| visualize_h5ad.py | Python 3 | ~350 | Analysis & visualization |

### Documentation Files

| File | Format | Purpose |
|------|--------|---------|
| README.md | Markdown | Project overview & guide |
| VCC_DATA_DESCRIPTION.md | Markdown | Detailed data documentation |
| h5ad_visualize.md | Markdown | Visualization guide |
| environment.yml | YAML | Conda environment |
| requirements.txt | Text | Python dependencies |

## Key Metadata

### H5AD Object Structure

```
adata
├── X               Sparse CSR matrix (221273 × 18080)
├── obs             Cell metadata (3 columns)
│   ├── target_gene     151 unique targets
│   ├── guide_id        189 unique guides
│   └── batch           48 unique batches
├── var             Gene metadata (1 column)
│   └── gene_id         Ensembl IDs
├── obs_names       Cell barcodes
├── var_names       Gene symbols
└── [uns, obsm, varm, obsp, varp] Empty
```

### Statistics Summary

| Category | Count/Value |
|----------|------------|
| Total Cells | 221,273 |
| Total Genes | 18,080 |
| Control Cells (non-targeting) | 38,176 (17.3%) |
| Perturbed Cells | 183,097 (82.7%) |
| Unique Target Genes | 151 |
| Unique Guide RNAs | 189 |
| Experimental Batches | 48 |
| Non-zero Elements | 1,932,554,688 |
| Expression Sparsity | 51.69% |
| Validation Target Genes | 50 |

## Workflow & Usage

### 1. Setup
```bash
cd /home/richard/projects/VCC
conda env create -f environment.yml -n vcc
conda activate vcc
```

### 2. Explore Data
```bash
python scripts/preview_data.py
python scripts/visualize_h5ad.py
```

### 3. Access Data
```python
import anndata as ad
adata = ad.read_h5ad('data/raw/adata_Training.h5ad')
```

### 4. Analyze & Process
- Create new scripts in `scripts/`
- Add notebooks in `notebooks/`
- Save outputs to `analysis/`

### 5. Document
- Update documentation in `docs/`
- Maintain this MANIFEST.md

## Best Practices

1. **Never modify raw data** - Keep `data/raw/` read-only
2. **Process data** - Save processed data to `data/processed/`
3. **Version control** - Use Git, respect `.gitignore`
4. **Document work** - Add markdown docs for new analyses
5. **Organize outputs** - Save results to appropriate `analysis/` subdirectories
6. **Name files clearly** - Use descriptive names for scripts and outputs

## Future Additions

```
vcc/
├── config/                      Configuration files
│   └── config.yaml
├── models/                      Trained models
│   └── (model checkpoints)
├── results/                     Final analysis results
│   ├── figures/
│   ├── tables/
│   └── reports/
└── tests/                       Unit tests
    └── test_*.py
```

---

**Last Updated:** November 12, 2025  
**Maintainer:** Richard  
**Status:** Active Development


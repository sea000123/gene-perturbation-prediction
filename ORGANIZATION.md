# VCC Repository Organization

## 📋 Summary

Your VCC project is now organized with a professional structure following best practices:

```
VCC/                               ← Project root
│
├── 📄 README.md                   ← START HERE - Main documentation
├── 📄 MANIFEST.md                 ← Detailed file inventory
├── 📄 ORGANIZATION.md             ← This file
├── 📦 requirements.txt             ← Python dependencies
├── 🔧 environment.yml              ← Conda environment config
├── .gitignore                      ← Git configuration
│
├── 📂 data/                        ← All data storage
│   ├── raw/                        ← Original dataset (DO NOT MODIFY)
│   │   ├── adata_Training.h5ad    ← 221K cells × 18K genes
│   │   ├── gene_names.csv
│   │   └── pert_counts_Validation.csv
│   └── processed/                  ← Future: processed data outputs
│
├── 🐍 scripts/                     ← Python analysis scripts
│   ├── preview_data.py             ← Quick data overview
│   └── visualize_h5ad.py           ← Generate plots & statistics
│
├── 📊 analysis/                    ← Analysis outputs
│   └── visualizations/             ← Generated plots (300 DPI)
│       ├── h5ad_visualization.png
│       └── h5ad_expression_details.png
│
├── 📚 docs/                        ← Documentation
│   ├── VCC_DATA_DESCRIPTION.md    ← Data specification & stats
│   └── h5ad_visualize.md          ← Visualization guide
│
└── 📓 notebooks/                   ← Jupyter notebooks (optional)
    └── (create as needed)
```

---

## 🎯 Quick Start

### 1️⃣ Setup Environment
```bash
cd /home/richard/projects/VCC
conda env create -f environment.yml -n vcc
conda activate vcc
```

### 2️⃣ Preview Data
```bash
python scripts/preview_data.py
```

### 3️⃣ Generate Visualizations
```bash
python scripts/visualize_h5ad.py
```

### 4️⃣ Read Documentation
- Start with `README.md` for overview
- See `docs/VCC_DATA_DESCRIPTION.md` for data details
- Check `MANIFEST.md` for complete file inventory

---

## 📁 Directory Guide

### 🟦 `/data` - Data Storage
- **`raw/`** - Original dataset files (read-only)
  - `adata_Training.h5ad` - Main gene expression data
  - `gene_names.csv` - Gene identifiers
  - `pert_counts_Validation.csv` - Validation metadata
  
- **`processed/`** - For processed/transformed data (future use)

### 🟦 `/scripts` - Analysis Scripts
- **`preview_data.py`** - Quick overview of all 3 files
- **`visualize_h5ad.py`** - Comprehensive analysis with plots

### 🟦 `/analysis` - Results & Outputs
- **`visualizations/`** - Generated plots and figures
  - `h5ad_visualization.png` - 6-panel summary
  - `h5ad_expression_details.png` - Expression matrix analysis

### 🟦 `/docs` - Documentation
- **`VCC_DATA_DESCRIPTION.md`** - Data format & statistics
- **`h5ad_visualize.md`** - Visualization methodology
- **`README.md`** - Project guide & usage
- **`MANIFEST.md`** - Complete file manifest

### 🟦 `/notebooks` - Analysis Notebooks (Optional)
- Add `.ipynb` files here for exploratory analysis
- Good for interactive data exploration

---

## 📊 Dataset at a Glance

| Metric | Value |
|--------|-------|
| **Cells** | 221,273 |
| **Genes** | 18,080 |
| **Target Genes** | 151 |
| **Control Cells** | 38,176 |
| **Experimental Batches** | 48 |
| **Expression Sparsity** | 51.69% |
| **Data Format** | AnnData H5AD (Sparse) |
| **File Size** | ~7.2 GB |

---

## 🔧 Configuration Files Explained

### `requirements.txt`
Python package versions for pip installation:
```bash
pip install -r requirements.txt
```

### `environment.yml`
Conda environment specification (recommended):
```bash
conda env create -f environment.yml -n vcc
```

### `.gitignore`
Prevents committing large data files and temporary files to Git

---

## 💡 Common Tasks

### Load & Explore Data
```python
import anndata as ad
import pandas as pd

adata = ad.read_h5ad('data/raw/adata_Training.h5ad')
print(adata)  # Overview
print(adata.obs.head())  # Cell metadata
```

### Add New Analysis
1. Create script in `scripts/` (e.g., `analyze_something.py`)
2. Run it: `python scripts/analyze_something.py`
3. Save outputs to `analysis/`
4. Document in `docs/`

### Add Jupyter Notebook
1. Create in `notebooks/` (e.g., `exploration.ipynb`)
2. Run: `jupyter notebook notebooks/exploration.ipynb`
3. Save results to `analysis/`

### Version Control
```bash
git add -A
git commit -m "Add analysis of expression patterns"
git push
```

---

## 🚀 Next Steps

1. **✅ Explore Data** - Run preview scripts
2. **✅ Read Docs** - Check documentation
3. **⚙️ Develop Models** - Build analysis pipelines
4. **📝 Add Notebooks** - Create analysis notebooks
5. **🔄 Share Results** - Document findings

---

## 📞 File Reference

| Need | File |
|------|------|
| Project overview | `README.md` |
| File inventory | `MANIFEST.md` |
| Setup instructions | `environment.yml` / `requirements.txt` |
| Data description | `docs/VCC_DATA_DESCRIPTION.md` |
| Visualization guide | `docs/h5ad_visualize.md` |
| Quick data preview | `scripts/preview_data.py` |
| Detailed analysis | `scripts/visualize_h5ad.py` |
| Visualization outputs | `analysis/visualizations/` |

---

## ✨ Best Practices

✅ **DO:**
- Keep `data/raw/` read-only
- Save processed data to `data/processed/`
- Document new analyses in `docs/`
- Use Git for version control
- Name files descriptively

❌ **DON'T:**
- Modify files in `data/raw/`
- Commit large data files
- Leave code undocumented
- Ignore `.gitignore` rules

---

**Status:** ✅ Repository Organized  
**Structure:** Production-Ready  
**Next Action:** Read `README.md` or run `scripts/preview_data.py`


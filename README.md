# IronClad Face Recognition System

This project evaluates key design trade-offs for deploying a face retrieval system capable of handling up to a billion gallery images, including model selection, preprocessing, indexing strategies, and deployment considerations.

## Project Structure

```
├── analysis/
│   ├── figures/             # Generated visualizations
│   ├── task1.ipynb          # Model selection (VGG vs CASIA)
│   ├── task2.ipynb          # Face detection preprocessing + indexing strategies
│   ├── task3.ipynb          # Top-N optimization
│   ├── task4.ipynb          # Gallery size optimization
│   └── task5.ipynb          # Uncertainty estimation
├── ironclad/                # Core system modules
├── multi_image_identities/  # Dataset (not included in repo)
├── CASE_ANALYSIS.md         # Full analysis report
└── README.md
```

## Key Findings

| Decision | Recommendation |
|----------|----------------|
| Embedding Model | VGGFace2 (70.9% vs 36.5% for CASIA) |
| Preprocessing | MTCNN face detection (+13.4% improvement) |
| Indexing | HNSW for billion-scale (6x faster, 1.5% accuracy trade-off) |
| Candidates Returned | N=5 for standard use, N=10 for high-security |
| Gallery Size | m=5 images per identity (VGG saturates here) |

## Running the Notebooks

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
cd analysis/
jupyter notebook
```

Notebooks should be run sequentially (task1 through task5) as later tasks depend on saved results from earlier ones.

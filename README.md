# IronClad Face Recognition System

This project evaluates key design trade-offs for deploying a face retrieval system capable of handling up to a billion gallery images, including model selection, preprocessing, indexing strategies, and deployment considerations.

## Project Structure

```
├── analysis/
│   ├── figures/                   
│   ├── task1.ipynb                
│   ├── task2.ipynb                
│   ├── task3.ipynb                
│   ├── task4.ipynb                
│   └── task5.ipynb                
├── ironclad/
│   ├── modules/                   # Embedding + retrieval implementations
│   ├── app.py
│   └── requirements.txt
├── test/                          # Unit tests
├── multi_image_identities/        # Dataset (not included in repo)
├── CASE_ANALYSIS.md               # Case analysis report
├── ANALYSIS_DESIGN_ASSIGNMENT.md
├── IMPLEMENTATION_ASSIGNMENT.md
├── setup.sh
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

# Graph-Based Intrusion Detection System (Developing Project)

This repository contains code and resources for implementing a Graph Neural Network (GNN) for intrusion detection in network systems. The project is a work-in-progress, exploring graph-based techniques such as E-GraphSAGE and Heterogeneous GNNs for node and edge classification tasks.

## Directory Overview

- **`feature_pics/`**:  
  A folder containing visual representations of features and graphs used during model development.

- **`GNN_implementation.ipynb`**:  
  Implements GNN-based models like E-GraphSAGE and Heterogeneous GNN (from models directory) for edge and node classification. The notebook also includes training and evaluation scripts for these models.

- **`classical_ML_implementation.ipynb`**:  
  Classical machine learning models such as Random Forest  - used from Kahraman Kostas Thesis on Anomaly detection using ML

- **`graph_creation.ipynb`**:  
  Handles the creation of graph structures (bipartite, tripartite) (type1 , type2) with node and edge embeddings, ensuring proper encoding of labels and features.

- **`preprocessing.ipynb`**:  
  Prepares raw data for graph construction, including cleaning, normalization, and formatting steps. Posses code that will sample the all_data into shorter files based on the number of rows as input

## Current Progress
- Data preprocessing and graph construction pipelines have been established.
- GNN architectures for edge and node classification are under testing.
- Initial results show promise, but model training on large datasets remains a bottleneck and is currently in progress.

## Next steps
- Expand to temporal GNN models for detecting zero-day attacks.
- Implement adversarial testing and robust evaluation metrics.




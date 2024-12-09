# Hypergraph-Modelling-using-DOSAGE
In this repository we propose a solution to the DOS problem via Agglomerative Greedy Enumeration (DOSAGE) algorithm as a novel approach to enhance the process of generating the densest overlapping subgraphs and, hence, a robust construction of the hypergraphs. Experiments on standard benchmarks show that the DOSAGE algorithm significantly outperforms the HGNNs and six other methods on the node classification task.

This is a code for Hyperedge Modeling in Hypergraph Neural Networks by using Densest Overlapping Subgraphs paper.
You can read the paper in here: https://arxiv.org/abs/2409.10340

## Setup Instructions

Follow these steps to set up the project locally.

### Prerequisites

- Python (>= 3.8)
- Conda (or Miniconda)

### Steps

1. Clone the repository:
   ```bash
   git clone hhttps://github.com/Mehrads/Hypergraph-Modelling-using-DOSAGE.git
   cd Hypergraph-Modelling-using-DOSAGE
   ```

Create a virtual environment:
   ```bash
   conda create --name env python=3.8
   ```

Activate the virtual environment:
   ```bash
   conda activate env
   ```

Install dependecies:
   ```bash
   conda install --file requirements.txt
   ```
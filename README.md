# Final Master's Project - Breast Lesion Classification with GREP and LLM-RAG

## Overview

This repository contains the code and scripts used in my final master's project, which focuses on classifying breast lesions from mammography reports using two different methods: GREP and LLM-RAG. The project aims to compare the performance and efficiency of these methods in extracting BI-RADS classifications and breast laterality from a large dataset of mammography reports.

## Contents

- `scripts/`: This directory contains all the scripts used for data preprocessing, model training, evaluation, and result visualization.
  - `grep_pipeline.sh`: Script for the GREP pipeline.
  - `llm_rag_pipeline.py`: Script for the LLM-RAG pipeline.
- `data/`: This directory should contain the mammography report dataset in CSV format.
- `results/`: This directory will store the output results and evaluation metrics.

## Getting Started

### Prerequisites

Ensure you have the following installed on your machine:

- Python 3.8 or higher
- Required Python packages listed in `requirements.txt`

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/Danidayscat/trabajo_final_de_master_UOC.git
```

### Navigate to the repository directory and install the required packages:

```bash
cd trabajo_final_de_master_UOC
pip install -r requirements.txt
```

## Results

The results and evaluation metrics are stored in the `results/` directory. These include the precision, recall, and F1-score for both the GREP and LLM-RAG pipelines.

## Acknowledgements

This project was developed as part of my final master's project at the Open University of Catalonia (UOC). Special thanks to my supervisors and the Parc de Salut Mar for providing the necessary resources and data for this study.



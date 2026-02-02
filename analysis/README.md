# SST Data Cleaning & SSRT Analysis Pipeline (Python)

This folder contains a Python-based analysis pipeline for cleaning behavioural data from a **Stop Signal Task (SST)** and computing key outcome measures, including **SSRT (Stop-Signal Reaction Time)**.

The goal of this pipeline is to demonstrate an **automated and reproducible** workflow for behavioural research data analysis (i.e., using own scripts rather than manual Excel cleaning).

---

## What this analysis does

The notebook performs the following steps:

1. **Batch-load participant files**
   - Automatically reads multiple PsychoPy `.csv` output files from a folder.

2. **Merge into one dataset**
   - Combines participant-level raw files into a single dataset for analysis.

3. **Clean behavioural data**
   - Removes non-trial rows
   - Converts key columns to numeric
   - Separates Go and Stop trials
   - Cleans Go RT data using standard thresholds:
     - removes RT < 200 ms
     - removes RT > 1250 ms (task maximum response window)

4. **Compute participant-level summary measures**
   - Go accuracy
   - Mean Go RT (correct Go trials only)
   - Stop success rate
   - Mean SSD
   - **SSRT (integration method)**

5. **Export outputs**
   - Saves clean/analysis-ready datasets as `.csv`

---

## Files in this folder

- `SST_data_cleaning.ipynb`  
  Main notebook for preprocessing and SSRT estimation.

- `requirements.txt`  
  Python dependencies required to run the notebook.

- `data_sample/`  
  Contains small anonymised example CSV files to demonstrate the pipeline.

---

## How to run

### 1) Install dependencies
```bash
pip install -r requirements.txt

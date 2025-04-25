# Classification-of-Antibiotic-Resistance-Gene-Sequences-using-Nucleotide-Transformer

Author – Yesasvi Sai Nandigam

Programming language & version – Python 3.10
Dependencies – Transformers, Datasets, Scikit-learn, Matplotlib, Pandas, NumPy
Files required – train.csv, val.csv, test.csv

Description

This Python project fine-tunes a pre-trained Nucleotide Transformer model to classify antibiotic resistance genes (ARGs) from DNA sequence data. It uses sequence data from the CARD and RefSeq databases, including synthetic non-resistant sequences, to train a robust model for genomic classification. The pipeline includes data preprocessing, training on an HPC cluster, and evaluation of accuracy, F1-score, and interpretability via attention-based architecture.

Execution

Data Preparation
	•	Extract and curate resistant, non-resistant, and synthetic non-resistant sequences
	•	Label data as 1 (ARG) and 0 (Not ARG), then split into train/validation/test sets

Model Setup
	•	Load InstaDeepAI/Nucleotide-Transformer-v2-500m-multi-species model from Hugging Face
	•	Tokenize sequences using k-mer batching and Hugging Face Datasets
	•	Train using Hugging Face Trainer API (CPU-optimized)

Training
	•	HPC-based training using Slurm, with batched tokenization, CPU optimization, and checkpointing
	•	Training done using 8 CPUs, 100GB memory, no GPU

Evaluation
	•	Confusion matrix
	•	Accuracy and F1 over epochs
	•	Training loss tracking
	•	Classification report (Precision, Recall, F1-score)

Output
	•	Confusion Matrix showing model prediction accuracy
	•	Line plot of Accuracy and F1 Score across training epochs
	•	Line plot of Training Loss over steps
	•	Classification Report with precision: 0.99, recall: 0.99, F1-score: 0.99

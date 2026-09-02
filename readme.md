# 🧬 DRTfold

## 1. Clone Code
```bash
git clone https://github.com/YangYiming00/DRTfold.git
```

## 2. Preparation
```bash
conda create -n DRTfold python=3.11.7
conda activate DRTfold
```

```bash
cd DRTfold
```

```bash
cd Protenix
pip install -e .
```

```bash
cd ..
cd DRfold2
bash install.sh
```

```bash
cd ..
cd rna_ss_reasoning
pip install -e . --use-deprecated=legacy-resolver
pip install pyarrow==12.0.1
```

The model checkpoint is available at [Hugging Face](https://huggingface.co/Yimingbear/DRTfold_ckpt).

## 3. Inference

### 3.1 Inference on Your Own RNA Sequence

Please change the corresponding paths in `main.sh` before running inference.

```bash
cd ..
bash main.sh
```

### 3.2 Reproducing the Results in the Paper

Download the benchmark datasets from:

- [RNA Secondary Structure Benchmarks](https://huggingface.co/datasets/Yimingbear/rna_ss_benchmark)
- [Geometric Priors](https://huggingface.co/datasets/Yimingbear/geometric_prior)

Then set the reasoning path:

```bash
cd rna_ss_reasoning/experiments/AIDO.RNA/rna_secondary_structure_prediction
bash rna_secondary_structure_prediction.sh test $BENCHMARK_NAME $MODEL_PATH
```

Supported benchmark names are:

```text
bpRNA
archiveII
PDB-TS1
PDB-TS2
PDB-TS3
Rfam15.0-15.1
bpRNA-new
PDB2607
```

Please make sure that `$MODEL_PATH` points to the downloaded model checkpoint before running the evaluation.


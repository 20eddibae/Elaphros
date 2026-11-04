# Elaphros

Elaphros is a small, fast sequence-to-function model. From 524,288 bp of DNA it predicts 4,670 genomic
tracks (ATAC, DNase, PRO-cap, CAGE, RNA-seq, TF ChIP and histone ChIP) at 128 bp resolution, and 1,854 of
them also at 1 bp.

| Model | Parameters | Sequence mixer |
|---|---|---|
| Elaphros | 50.3M | 14 bidirectional Hydra SSM blocks at 32 bp |
| Elaphros-Conv | 49.3M | 12 dilated convolution blocks at 128 bp |

Both are U-Nets. A Cheri-block convolution encoder (1 to 32 or 64 bp) feeds the mixer. A decoder with skip
connections returns to 1 bp. Low-rank linear heads read a 128 bp and a 1 bp embedding.

An optional **cell branch** predicts tracks for a cell type never seen in training from that cell's measured
accessibility (one ATAC or DNase track). It uses position-specific FiLM and, optionally, conditional LoRA.

## Install

```bash
pip install -e .                    # torch, numpy, pandas, zstandard
pip install -e ".[preprocessing]"   # pyBigWig, pysam (data pipeline)
pip install mamba-ssm               # optional: fast Hydra scan on GPU (a pure-PyTorch scan is used otherwise)
pip install borzoi-pytorch enformer-pytorch   # optional: comparators
```

```python
import torch
from elaphros import Elaphros, elaphros_config, load_checkpoint

model = Elaphros(elaphros_config()).eval()             # or: model, step = load_checkpoint("snap_step060000.pt")
x = torch.nn.functional.one_hot(torch.randint(0, 4, (1, 524288)), 4).float()
out = model(x, heads_1bp=False)                        # {"dnase@128": (1, 384, 4096), ...}
```

Each head has a fixed number of output slots. Track `j` of an assay family is slot `j`, in the order of the
data root's `tracks.tsv`. A checkpoint is read with the `tracks.tsv` of the data it was trained on.

## Repository

```
elaphros/        model, Hydra SSM, cell branch, data loading, losses, Muon optimizer, comparator wrappers
scripts/         train.py, train_cellcond.py, finetune_utr.py, evaluate_tracks.py,
                 evaluate_cellcond.py, benchmark_speed.py
preprocessing/   ENCODE track selection, FANTOM5 CAGE, download and conversion, crop plan, cache assembly
```

## How the RECOMB 2027 submission was run

All models were trained from scratch on one A100, one seed per model.

### 1. Data

Inputs:
- GRCh38 reference (UCSC hg38) with its `.fai` index.
- Borzoi's 8-fold split, `sequences_human.bed.gz` from calico/borzoi: folds 0, 1, 2, 5, 6 and 7 train,
  fold 3 validates, and fold 4 tests.
- ENCODE GRCh38 files.
- FANTOM5 hg38 CTSS files (CC BY 4.0).

```bash
python preprocessing/select_tracks.py  --work W                                  # ENCODE portal -> W/manifest.tsv
python preprocessing/fantom_cage.py    --work W --libraries fantom_libraries.tsv # + FANTOM5 CAGE columns
python preprocessing/make_crops.py     --work W --genome hg38.fa --folds sequences_human.bed.gz
python preprocessing/process_tracks.py --work W --genome hg38.fa --shard 0/8     # one job per shard
python preprocessing/assemble.py       --work W --genome hg38.fa --folds sequences_human.bed.gz --out DATA
```

How each assay becomes a track:
- **Selection:** one track per biosample (or per target and biosample, or per assay and biosample), from
  released, unrestricted ENCODE-consortium experiments with no genetic modification, no treatment and no
  ERROR audit.
- **ATAC and DNase:** 1 bp cut-site counts from filtered BAMs, with 5′ ends shifted +4/−4 (ATAC) or 0/+1
  (DNase).
- **ChIP:** fold change over control, averaged per 128 bp.
- **ENCODE CAGE, RNA-seq and PRO-cap:** stranded signal of unique reads.
- **FANTOM5 CAGE:** CTSS tag counts from 296 primary-cell and tissue libraries (592 stranded columns). The
  selection rule is described in `preprocessing/fantom_cage.py`, which takes the chosen library list as input.
- **Normalisation:** each track is divided by its mean non-zero value over training regions.

`DATA` then holds:
- the reference;
- the fold file;
- `tracks.tsv`;
- a genome-wide 128 bp cache;
- 32,768 bp crops of the 1 bp tracks, one centred in every training interval and in 400 intervals each of
  folds 3 and 4.

### 2. Pretraining (Elaphros and Elaphros-Conv)

```bash
python scripts/train.py --data DATA --out runs/elaphros      --model elaphros
python scripts/train.py --data DATA --out runs/elaphros_conv --model elaphros-conv
```

The defaults are the submission recipe:
- **Schedule:** 60,000 steps over a context curriculum of 64 kb × 8, then 131 kb × 4, then 524 kb × 1
  (40/40/20% of steps; 512 kb of sequence per step).
- **Loss:** Poisson NLL of 8,192 bp segment totals plus 5 × the multinomial NLL of positions within each
  segment.
- **Targets:** divided by the track normaliser, RNA raised to the power 0.75, values above 10 soft-clipped.
- **Optimiser:** Muon for block weights and AdamW for the rest, both at learning rate 8e-4, with 2,000 warmup
  steps and cosine decay.
- **Other:** EMA 0.999; reverse-complement augmentation with p = 0.5; half of the 1 bp tracks sampled per
  step; bf16.

Training took 8.5 h for Elaphros and 5.75 h for Elaphros-Conv. Results use the EMA weights at step 60,000
(`snap_step060000.pt`).

### 3. Cell-state conditioning

```bash
python scripts/train_cellcond.py --data DATA --split heldout.json --out runs/film                       # FiLM
python scripts/train_cellcond.py --data DATA --split heldout.json --out runs/film_lora --lora-ranks 1,2,5,10 --lora-scale 1.0
python scripts/train_cellcond.py --data DATA --split heldout.json --out runs/lora002   --lora-ranks 1,2,5,10 --lora-scale 0.02
python scripts/train_cellcond.py --data DATA --split heldout.json --out runs/film_pool --pool                # global FiLM
python scripts/train_cellcond.py --data DATA --split heldout.json --out runs/lora_only --lora-ranks 1,2,5,10 --film-freeze
python scripts/train.py --data DATA --heldout-split heldout.json --steps 40000 --out runs/base   # Base, no branch
python scripts/evaluate_cellcond.py --data DATA --split heldout.json --ckpt runs/film/snap_step040000.pt \
    --biosamples test --out results/film_test.json
python scripts/evaluate_cellcond.py ... --baseline borzoi --targets borzoi_targets_human.txt --out results/film_vs_borzoi.json
```

- **Held-out cells:** 30 biosamples (10 VAL, 20 TEST); every one of their tracks is removed from all losses.
  `make_split` writes a split file if none exists; the submission used a fixed split file.
- **Training:** base model and branch train together for 40,000 steps with the pretraining recipe. Each step
  draws 8 training biosamples per window and supervises up to 24 of each one's tracks through the branch.
- **Size:** the branch adds 2.2M parameters, and LoRA 0.8M more.
- **Scoring:** partial Pearson of the prediction with the truth, controlling for the training-cell mean track,
  on the central 131 kb of fold 4 windows for TEST cells. Bars on the same rows:
  - the nearest training cell's measured track;
  - the cell's own accessibility;
  - the model's sequence-only head;
  - a random training cell's accessibility as input.
- **Comparators:** `--baseline` scores Borzoi or Enformer on the same rows using their channel for the nearest
  training cell (NEAREST), the mean of their same-assay channels (MEAN), or their own channel for the cell
  (EXACT, a ceiling).
- **Base (unconditioned) model:** `train.py --heldout-split` removes the held-out biosamples from the loss.
  `evaluate_cellcond.py` on its checkpoint reports Base as `bar_d`, its head for the nearest training cell.

### 4. Fine-tuning on a 5′UTR MPRA

```bash
python scripts/finetune_utr.py --csv GSM3130435_egfp_unmod_1.csv --ckpt runs/elaphros/snap_step060000.pt \
    --arms bar_kmer,cnn_small,probe,full_ft,scratch,borzoi_probe,borzoi_ft,enformer_probe,enformer_ft \
    --budgets 1024,16384 --seeds 0,1 --out results/utr
```

- **Data:** Sample et al. 2019 (GEO GSE114002); the target is mean ribosome load.
- **Splits:** fixed test (20,000 rows) and selection (10,000 rows) sets; each budget is a prefix of one
  shuffled training pool.
- **Input:** each 50 bp insert is centred in a fixed 512 bp flank.
- **Arms:**
  - k-mer ridge;
  - a 0.47M CNN trained from scratch;
  - ridge probes on frozen features;
  - full fine-tuning of Elaphros, Borzoi and Enformer (Enformer with BatchNorm statistics frozen);
  - Elaphros from random initialisation.
- **Selection:** each arm's learning rate or ridge penalty is chosen on the selection split, and the selected
  model is scored on test (R²).

### 5. Track evaluation and speed

```bash
python scripts/evaluate_tracks.py --data DATA --model elaphros --ckpt runs/elaphros/snap_step060000.pt \
    --match-targets borzoi_targets_human.txt --gtf gencode.v48.basic.annotation.gtf --out results/elaphros.json
python scripts/evaluate_tracks.py --data DATA --model borzoi   --targets borzoi_targets_human.txt --gtf ... --out results/borzoi.json
python scripts/evaluate_tracks.py --data DATA --model flashzoi --targets borzoi_targets_human.txt --gtf ... --out results/flashzoi.json
python scripts/evaluate_tracks.py --data DATA --model enformer --targets enformer_targets_human.txt --span 32768 \
    --exclude-bed enformer_train.bed --out results/enformer.json                     # Elaphros: same --span / --exclude-bed
python scripts/benchmark_speed.py variant --model elaphros --precision fp32 --compile
python scripts/benchmark_speed.py length  --model elaphros --compile        # also --model borzoi / enformer
```

- **Track scoring:** comparators are scored on our tracks that share an experiment accession, assay and
  strand with one of their outputs.
- **Metric:** per-track Pearson over all 128 bp bins of the central 196,608 bp of every fold 3 and fold 4
  window, on raw values and on log1p values after a per-track scale fitted on the other fold. Scores are
  averaged within each assay, then across assays.
- **Gene level:** RNA exon sums and CAGE TSS sums, log2(x + 1) Pearson across genes.
- **Enformer panel:** uses the central 32 kb of windows clear of Enformer's training regions.
- **Speed:** a variant is a REF and an ALT forward with the 128 bp heads, at native input length. The length
  sweep times the forward pass at 131 kb, 524 kb and 1 Mb.

Inputs not produced by this repo:
- `heldout.json`: the held-out biosample split, `{"val": [...], "test": [...]}` of ontology term ids.
- `borzoi_targets_human.txt`: `examples/targets_human.txt` from calico/borzoi.
- `enformer_targets_human.txt`: `manuscripts/cross2020/targets_human.txt` from calico/basenji.
- `enformer_train.bed`: Enformer's training intervals, from Basenji's human `sequences.bed`.
- `gencode.v48.basic.annotation.gtf`: from GENCODE. Protein-coding genes on basic transcripts are used,
  excluding readthrough genes.

Comparators are used for evaluation only, from public weights:
- Borzoi: `johahi/borzoi-replicate-0`
- Flashzoi: `johahi/flashzoi-replicate-0`, with a plain-PyTorch attention port for accuracy
- Enformer: `EleutherAI/enformer-official-rough`

Target tables come from calico/borzoi and calico/basenji.

## Acknowledgements

- Hydra is adapted from calico/baskerville (Apache-2.0), which builds on mamba_ssm (Apache-2.0); see `NOTICE`.
- Cheri blocks follow Cherimoya.
- Data: ENCODE Consortium and FANTOM5 (RIKEN, CC BY 4.0).

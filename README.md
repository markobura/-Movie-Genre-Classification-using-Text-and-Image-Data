# Movie genre classification (text + posters)

Multi-label classification of 13 MovieScope genres from **plot synopses** and **posters**, then fusion of the two. Reference: LeBaron, Stanford CS230, [Multi-label Movie Genre Classification Using Multiple Modalities](http://cs230.stanford.edu/projects_fall_2021/reports/102983714.pdf).

The paper uses four modalities: poster, video, metadata, and text. We use plot synopses and posters. Video is not in the dump as frames or clips, so it was not practical to train. Metadata is in a CSV (budget, year, IMDb score, and similar); in the paper it is the weakest modality (0.457 macro AP), so we skipped it. Plot text and poster images are complete for every split, and that is the pair we train on.

Splits: **3454 / 491 / 989**. Primary metric: **macro Average Precision**.

### Models

| Part | Model | Role |
|---|---|---|
| Text | GloVe mean + MLP | paper replica |
| Text | TF-IDF + OneVsRest logistic regression | **best text**, used in fusion |
| Text | MiniLM embeddings + LogReg | extra text baseline |
| Poster | VGG-16, frozen conv (paper setup) | paper replica |
| Poster | VGG-16 + dropout, augmentations, weight decay | overfitting check |
| Poster | CLIP ViT-B/32 frozen encoder + linear head | **best poster**, used in fusion |
| Fusion | Mean of CLIP and TF-IDF probabilities | **reported multimodal result** |
| Fusion | Linear layer on concatenated probabilities | paper-style late fusion |

Notebooks (read these for the full argument):

- `notebooks/01_text_classification.ipynb`
- `notebooks/02_poster_classification.ipynb`
- `notebooks/03_fusion.ipynb`

## Results (test)

| Model | macro AP | micro AP | sample AP |
|---|---|---|---|
| Mean fusion (poster + text) | **0.7479** | 0.7915 | 0.8618 |
| Text, TF-IDF + LogReg | 0.7268 | 0.7733 | 0.8430 |
| Late fusion (paper-style linear head) | 0.7066 | 0.7551 | 0.8332 |
| Poster, CLIP ViT-B/32 | 0.6175 | 0.6219 | 0.7416 |
| Poster, VGG-16 (paper replica) | 0.4545 | 0.4776 | 0.6000 |
| Paper, poster | 0.4463 | 0.5201 | 0.6572 |
| Paper, text | 0.6195 | 0.6317 | 0.7497 |
| Paper, combined (4 modalities) | 0.5641 | 0.6270 | 0.7358 |

We report **mean fusion** (0.7479). Averaging poster and text probabilities beats both unimodal models and the trained late-fusion layer (0.7066). The layer overfits on 491 val examples (0.7797 fit AP vs 0.7066 test). Poster gain is concentrated on visual genres (`animation` +0.152 over text, also `comedy` and `family`); on `biography` the poster hurts.

Limitation: their Combined score uses four modalities, ours uses two, so those numbers are not the same comparison. Same-modality: CLIP poster 0.6175 vs paper poster 0.446, TF-IDF text 0.7268 vs paper GloVe MLP 0.6195.

## Setup

Python 3.11+ (this repo was run with 3.14). On macOS, `--device mps`; on NVIDIA, `--device cuda`; otherwise `--device cpu`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Data

Expected layout after the MovieScope split:

```
train_data/  val_data/  test_data/     # posters + {split}_labels.csv
data/movie_scope/                      # original dump
data/glove/glove.6B.300d.txt           # only for the GloVe replica in notebook 01
```

If you still have the raw MovieScope dump and no split folders:

```bash
python scripts/split_dataset.py
```

`train_data/`, `val_data/`, `test_data/`, `models/`, and `results/` are gitignored (large).

## Reproduce text

Run `notebooks/01_text_classification.ipynb` top to bottom with the `.venv` kernel.

What it trains:

1. GloVe mean + MLP (paper replica; needs `data/glove/glove.6B.300d.txt`)
2. TF-IDF + OneVsRest logistic regression (**best text model**)
3. MiniLM embeddings + LogReg

Section 11 writes `models/text_predictions.npz` (needed for fusion) plus sklearn dumps under `models/`.

## Reproduce posters

```bash
python scripts/train_poster.py --backbone vgg16_stanford --device mps --patience 10
python scripts/train_poster.py --backbone vgg16_regularized --device mps
python scripts/train_poster.py --backbone clip_vit_b32 --device mps --batch-size 16

python scripts/evaluate.py --backbone clip_vit_b32 --device mps --batch-size 16
```

Checkpoints go to `models/poster/<backbone>/best.pt`. Test metrics go to `results/poster/`. Walkthrough: `notebooks/02_poster_classification.ipynb`.

CLIP is a frozen encoder plus a linear head. Best checkpoint is epoch 1 (the head overfits after that). Early stopping is on **val macro AP**, not val loss.

## Reproduce fusion

Needs text `npz` from notebook 01 and CLIP val + test probability CSVs:

```bash
python scripts/predict.py --backbone clip_vit_b32 --split-dir val_data --device mps --batch-size 16
python scripts/predict.py --backbone clip_vit_b32 --split-dir test_data --device mps --batch-size 16
python scripts/fuse_predictions.py --device cpu
```

Writes `results/fusion/comparison_test.csv` and `test_metrics.json`. Walkthrough: `notebooks/03_fusion.ipynb`.

Fusion is trained on **validation** predictions, not train (base models are overfit on train). Default late fusion: 5000 full-batch Adam steps.

## Layout

```
helpers/                 shared genres, splits, AP, paper numbers
src/poster/              VGG-16 and CLIP models, datasets, metrics
src/fusion/              load aligned predictions, late fusion head
scripts/                 train / evaluate / predict / fuse / split
notebooks/               01 text, 02 poster, 03 fusion
```

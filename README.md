# Automated Authentication of Van Gogh Paintings
### Through Image Processing and Feature Engineering

**ARTI 403 – Image Processing · Group 3**
College of Computer Science & Information Technology, Imam Abdulrahman Bin Faisal University

---

## Overview

This project decides whether a painting is an **authentic van Gogh** or a **counterfeit / non-van Gogh** work. Each painting is cleaned and enhanced with image-processing techniques, described by hand-crafted colour, texture and edge features, and then classified with machine-learning models.

- **Task:** binary classification. Label `0` = authentic (van Gogh), label `1` = counterfeit / non-van Gogh.
- **Data:** 430 paintings: 124 authentic van Goghs, and 306 others (207 paintings by other artists, 9 Otto Wacker forgeries, 30 Stable Diffusion images and 60 StyleGAN3 images).
- **Features:** 137 numeric features per painting (colour, intensity, LBP, GLCM, edges / morphology, stroke orientation).
- **Models:** SVM, logistic regression, Random Forest, histogram gradient boosting, and a soft-voting ensemble, compared against a majority-class baseline.

```
paintings ─► resize + blur + CLAHE ─► 4×4 patches ─► edges + morphology ─► 137 features / painting
                                                                                  │
                                              work/features/painting_features.csv ◄┘
                                                                                  │
   evaluation, ablations, feature importance ◄─ SVM · LogReg · RF · HistGB · Ensemble
```

## Repository layout

```
.
├── ARTI403_VanGogh_Authentication.ipynb   # the whole pipeline: Part 1 (pre-processing) + Part 2 (machine learning)
├── README.md
├── requirements.txt
├── vgdb_2016/                             # VGDB-2016 dataset (download, see below)
│   ├── train/{vg,nvg}/                    #   vg = van Gogh, nvg = other artists
│   ├── test/{vg,nvg}/
│   ├── check/                             #   2 unlabelled images, skipped in the notebook
│   ├── vgdb_2016.csv, license.txt, readme.txt
├── Vincent Forgeries Wacker/              # Van Gogh Authentication Dataset (download, see below)
├── Vincent Stable Diffusion/
├── Vincent GAN finetune/
├── Vincent GAN random/
├── van_gogh_forgeries.csv                 # Not read by the notebook, available in the repo for viewing
├── work/                                  # created by Part 1 (images, patches, edge maps, feature table)
└── work_ml/                               # created by Part 2 (saved models, result tables)
```

`work/` and `work_ml/` are generated and can be deleted at any time. Re-running the notebook recreates them.

## Datasets

The datasets are not redistributed with the code. Download them and place them in the folders shown above. The folder names must match exactly for the notebook to run normally, otherwise you can edit `SOURCES` / `VGDB` in the Configuration cell to import them the way you like.

| Dataset | Contents | Link |
|---|---|---|
| **VGDB-2016** (Folego, Gomes & Rocha, ICIP 2016) | 126 van Gogh paintings and 207 paintings by about 100 other artists, as PNG files up to 23,235 × 20,366 px | https://dx.doi.org/10.6084/m9.figshare.3370627 |
| **Vincent van Gogh Authentication Dataset** (Schaerf, Ostmeyer, Popovici & Buividovich, 2023; CC BY 4.0) | 9 Otto Wacker forgeries, 30 Stable Diffusion images, 30 fine-tuned and 30 random StyleGAN3 images | https://zenodo.org/records/10276928 |

How the notebook labels the images:

| Subtype | Label | Where it comes from |
|---|---|---|
| `van_gogh` | 0 | `vgdb_2016/{train,test}/vg` |
| `other_artist` | 1 | `vgdb_2016/{train,test}/nvg` |
| `wacker_forgery`, `stable_diffusion`, `stylegan3_finetuned`, `stylegan3_random` | 1 | the four `Vincent ...` folders (every image in the folder gets the folder's label) |


## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook ARTI403_VanGogh_Authentication.ipynb
```

## Running the notebook

The notebook has two parts. Part 2 only needs the feature table written by Part 1, so the two parts can be run separately.

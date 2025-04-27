# Adversarial-Attack Ablation Study

This repository reproduces a systematic study of **embedding-space adversarial attacks** on NLP models.  
The original monolithic script (`ablation_multi.py`) has been **split into a clean, importable package** while preserving every line of computational logic.

---

## 1 · Introduction
We investigate how three independent design axes—**attack algorithm, loss function, and regularisation penalty**—affect attack-success-rate (ASR) on two sentiment-analysis classifiers:

* **DistilBERT** (Hugging Face SST-2 fine-tune)  
* **Bi-LSTM** (TextAttack SST-2 checkpoint)

Perturbations are performed **directly in embedding space**; the input text itself is not edited.

---

## 2 · Problem Statement & Methods
For a correctly classified sentence \(\mathbf{x}\) with label \(y\), we seek a minimal perturbation \(\delta\) in hidden-state space such that  

\[
\arg\max f(\mathbf{h}+\delta)=1-y,
\]

where \(f\) is the classifier head and \(\mathbf{h}\) is the clean embedding.

We sweep:

| **Dimension** | **Values** |
|---------------|------------|
| **Attack**    | Rotation [^1], PGD, DeepFool |
| **Loss**      | Cross-Entropy (CE), Triplet, Carlini–Wagner (CW) |
| **Penalty**   | KL, L<sub>2</sub>, Cosine |
| **Model**     | DistilBERT, LSTM |

[^1]: *RotationAttack* learns two orthogonal axes \(\mathbf{u},\mathbf{v}\) and a rotation angle \(\theta\) inside their span.

---

## 3 · Directory Layout



---

## 4 · Installation
```bash
git clone https://github.com/<your-handle>/adv-attacks-ablation.git
cd adv-attacks-ablation
python -m venv .venv && source .venv/bin/activate   # or conda env create
pip install -r requirements.txt
```

---

## 5 · Usage
### 5.1 DistilBERT sweep
```bash
python ablation_multi.py \
  --models DistilBERT \
  --attacks Rotation PGD DeepFool \
  --losses CE CW \
  --penalties KL L2 Cosine \
  --lambdas 8 0.6 200 10 1 200 \
           8 0.7 100 10 1 200 \
           0.0 0.0 \
  --dataset_size 50 \
  --pgd_steps 500  --pgd_alpha 0.01 \
  --rotation_steps 300 --rotation_lr 0.005 \
  --deepfool_steps 1000
```

###5.2 LSTM sweep
```bash
python ablation_multi.py \
  --models LSTM \
  --attacks Rotation DeepFool \
  --losses CE CW \
  --penalties KL L2 Cosine \
  --lambdas 1 0.4 75 0.01 0.1 10 \
           0.0 0.0 \
  --dataset_size 50 \
  --pgd_steps 500  --pgd_alpha 0.01 \
  --rotation_steps 300 --rotation_lr 0.005 \
  --deepfool_steps 1000
```

## 6. Results
Model        | Attack   | Loss    | Penalty  | λ          | ASR  (%)
-------------+----------+---------+----------+------------+---------
DistilBERT   | Rotation | CE      | KL       | 8          | 42.00
...

## 7 · License
Released under the MIT License (see LICENSE).

## 8 · Acknowledgments
Hugging Face Transformers & Datasets

TextAttack LSTM SST-2 baseline

PyTorch for the deep-learning stack









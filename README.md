# Game Of Adversaries

This repository cotains the code for a systematic study of **embedding-space adversarial attacks** on two SST-2 sentiment classifiers (a HuggingFace DistilBERT model and a TextAttack LSTM). It generates adversarial examples in embedding space via each attack/loss/penalty combo, measures attack success rates (ASR), and prints a summary table of ASR (%) for every configuration and measures the semantic similarity using a LLM judge (GPT-2).

---

## 1 · Introduction
We investigate how three independent design axes—**attack algorithm, loss function, and regularisation penalty**—affect attack-success-rate (ASR) on two sentiment-analysis classifiers:

* **DistilBERT** (Hugging Face SST-2 fine-tune)  
* **Bi-LSTM** (TextAttack SST-2 checkpoint)

Perturbations are performed **directly in embedding space**; the input text itself is not edited.

---

## 2 · Problem Statement & Methods
For a correctly classified sentence, we seek a minimal perturbation in hidden-state space such that the it leads to missclassication.


We cover:

| **Dimension** | **Values** |
|---------------|------------|
| **Attack**    | Rotation, PGD, DeepFool |
| **Loss**      | Cross-Entropy (CE), Carlini–Wagner (CW) |
| **Penalty**   | KL, L<sub>2</sub>, Cosine |
| **Model**     | DistilBERT, LSTM |

For each **successful** adversarial example we query an auxiliary language model (GPT-2) asking it to rate the similarity between original and perturbed embedded and find the average for each combination.


---

## 3 · Directory Layout


```text
adv-attacks-ablation/
├── README.md
├── requirements.txt
├── LICENSE
│
│
├── src/                    
│   ├── __init__.py
│   ├── config.py           
│   ├── runner.py           # main experiment loop
│   │
│   ├── judge.py            # GPT-2 semantic-similarity scorer
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── distilbert.py
│   │   └── lstm.py
│   │
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── rotation.py
│   │   ├── pgd.py
│   │   └── deepfool.py
│   │
│   ├── losses/
│   │   ├── __init__.py     # loss_map
│   │   ├── ce.py
│   │   ├── triplet.py
│   │   └── cw.py
│   │
│   └── penalties/
│       ├── __init__.py     # pen_map
│       ├── kl.py
│       ├── l2.py
│       └── cosine.py
│
└── ablation_multi.py      
```

---

## 4 · Installation
```bash
git clone https://github.com/ajayjagan2511/game_of_adversaries.git
cd game_of_adversaries
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

### 5.2 LSTM sweep
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
![image](https://github.com/user-attachments/assets/977c62be-e8d3-437f-bf16-13819ee9a7ce)

...

## 7 · License
Released under the MIT License (see LICENSE).

## 8 · Acknowledgments

Dr. Kuan-Hao Huang, Texas A&M University

Hugging Face Transformers & Datasets

TextAttack LSTM SST-2 baseline

PyTorch for the deep-learning stack









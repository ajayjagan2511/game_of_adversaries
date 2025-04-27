import argparse

def get_args():
    p = argparse.ArgumentParser("Ablation: DistilBERT & LSTM")
    p.add_argument("--models",   nargs="+", required=True,
                   choices=["DistilBERT", "LSTM"])
    p.add_argument("--attacks",  nargs="+", required=True,
                   choices=["Rotation", "PGD", "DeepFool"])
    p.add_argument("--losses",   nargs="+", required=True,
                   choices=["CE", "Triplet", "CW"])
    p.add_argument("--penalties", nargs="+", required=True,
                   choices=["KL", "L2", "Cosine"])
    p.add_argument("--lambdas",  nargs="+", type=float, required=True,
                   help="One λ per combo; DeepFool λ’s ignored")
    p.add_argument("--dataset_size",   type=int,   default=20)
    p.add_argument("--pgd_steps",      type=int,   default=100)
    p.add_argument("--pgd_alpha",      type=float, default=1e-2)
    p.add_argument("--rotation_steps", type=int,   default=100)
    p.add_argument("--rotation_lr",    type=float, default=1e-2)
    p.add_argument("--deepfool_steps", type=int,   default=1000)
    return p.parse_args()

#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# silence CUDA / XLA alerts
os.environ["TF_CPP_MIN_LOG_LEVEL"]          = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"

# local package imports
from .config    import get_args
from .models    import load_model
from .attacks   import RotationAttack, PGDAttack, DeepFoolAttack
from .losses    import loss_map
from .penalties import pen_map

curr_model = None
neg_means  = {}

def main():
    global curr_model, neg_means

    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) (model, attack, loss, penalty) combos
    combos = []
    for m in args.models:
        for atk in args.attacks:
            if atk == "DeepFool":
                for loss in args.losses:
                    combos.append((m, atk, loss, None))
            else:
                for loss in args.losses:
                    for pen in args.penalties:
                        combos.append((m, atk, loss, pen))
    if len(combos) != len(args.lambdas):
        raise ValueError(f"need {len(combos)} λ’s, got {len(args.lambdas)}")

    # 2) load victim models
    victims = {m: load_model(m, device) for m in args.models}

    # 3) introspect LSTM layers
    for info in victims.values():
        if info["type"] == "ta":
            mdl = info["model"]
            info["emb_layer"]  = next(m for m in mdl.modules()
                                      if isinstance(m, nn.Embedding))
            info["lstm_layer"] = next(m for m in mdl.modules()
                                      if isinstance(m, nn.LSTM))
            info["fc_layer"]   = next(m for m in mdl.modules()
                                      if isinstance(m, nn.Linear))

    # 3.5) pre-compute mean negative-class embeddings for triplet loss
    train_ds = load_dataset("glue", "sst2", split="train")
    for m_name, info in victims.items():
        model = info["model"]
        per_label = {0: [], 1: []}
        if info["type"] == "hf":
            for ex in train_ds:
                txt, lbl = ex["sentence"], ex["label"]
                inp = info["tokenizer"](txt, return_tensors="pt",
                                         truncation=True, padding=True).to(device)
                with torch.no_grad():
                    emb = info["encoder"](**inp).last_hidden_state[:, 0, :].detach()
                per_label[lbl].append(emb.squeeze(0))
        else:
            for ex in train_ds:
                txt, lbl = ex["sentence"], ex["label"]
                ids = info["tokenizer"](txt)
                inp_ids = torch.tensor(ids, device=device).unsqueeze(0)
                with torch.no_grad():
                    seq_emb = info["emb_layer"](inp_ids.t())
                    _, (h_o, _) = info["lstm_layer"](seq_emb)
                    h_last = torch.cat((h_o[-2], h_o[-1]), 1).detach().squeeze(0)
                per_label[lbl].append(h_last)
        neg_means[model] = {
            0: torch.mean(torch.stack(per_label[0]), 0, keepdim=True),
            1: torch.mean(torch.stack(per_label[1]), 0, keepdim=True),
        }

    dataset = load_dataset("glue", "sst2",
                           split=f"validation[:{args.dataset_size}]")

    # 4) ablation sweep (identical to original logic)
    for ((m_name, atk, loss_name, pen_name), lam) in zip(combos, args.lambdas):
        info       = victims[m_name]
        curr_model = info["model"]
        success = total = 0

        curr_model.train() if info["type"] == "ta" else curr_model.eval()

        for ex in dataset:
            txt, lbl = ex["sentence"], ex["label"]
            emb = None

            # DistilBERT path
            if info["type"] == "hf":
                with torch.no_grad():
                    inp  = info["tokenizer"](txt, return_tensors="pt").to(device)
                    emb  = info["encoder"](**inp).last_hidden_state[:, 0, :].detach()
                    pred = curr_model.classifier(emb).argmax(-1).item()
            else:
                # LSTM path
                ids     = info["tokenizer"](txt)
                inp_ids = torch.tensor(ids, device=device).unsqueeze(0)
                with torch.no_grad():
                    seq_emb = info["emb_layer"](inp_ids.t())
                    _, (h_o, _) = info["lstm_layer"](seq_emb)
                    h_cat = torch.cat((h_o[-2], h_o[-1]), 1)
                    pred  = info["fc_layer"](h_cat).argmax(-1).item()

            if pred != lbl:
                continue
            total += 1
            y = torch.tensor([1 - lbl], device=device)

            # ---------- attacks on DistilBERT ----------
            if emb is not None:
                if atk == "Rotation":
                    atk_mod  = RotationAttack(emb.size(-1)).to(device)
                    optim = torch.optim.Adam(atk_mod.parameters(),
                                             lr=args.rotation_lr)
                    for _ in range(args.rotation_steps):
                        adv = atk_mod(emb)
                        clf, _ = loss_map[loss_name](emb, adv, y)
                        penv = 0.0 if pen_name is None \
                               else lam * pen_map[pen_name](emb, adv)
                        beta = -1 if loss_name == "Triplet" else 1
                        (beta*clf + penv).backward()
                        optim.step(); optim.zero_grad()
                    out_lbl = curr_model.classifier(adv).argmax(-1).item()
                    if out_lbl == 1 - lbl: success += 1

                elif atk == "PGD":
                    pen_fn = (lambda o,a: 0.0) if pen_name is None \
                             else pen_map[pen_name]
                    adv = PGDAttack(args.pgd_alpha, args.pgd_steps, lam)(
                            emb, loss_map[loss_name], pen_fn, y)
                    out_lbl = curr_model.classifier(adv).argmax(-1).item()
                    if out_lbl == 1 - lbl: success += 1

                else:  # DeepFool
                    adv = DeepFoolAttack(args.deepfool_steps)(emb)
                    out_lbl = curr_model.classifier(adv).argmax(-1).item()
                    if out_lbl == 1 - lbl: success += 1

            # ---------- attacks on LSTM ----------
            else:
                seq_len, _, emb_dim = seq_emb.shape
                flat_o = seq_emb.permute(1,0,2).reshape(1, -1)

                def loss_seq(o_flat, a_flat, yv):
                    o_seq = o_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                    _, (h_o, _) = info["lstm_layer"](o_seq)
                    h_cat_orig = torch.cat((h_o[-2], h_o[-1]), 1)
                    a_seq = a_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                    _, (h_a, _) = info["lstm_layer"](a_seq)
                    h_cat_adv  = torch.cat((h_a[-2], h_a[-1]), 1)
                    logits = info["fc_layer"](h_cat_adv)
                    if loss_name == "CE":
                        return F.cross_entropy(logits, yv), logits
                    from .losses.triplet import _triplet
                    neg = neg_means[curr_model][yv.item()].to(h_cat_adv.device)
                    return _triplet(h_cat_adv, neg, h_cat_orig), logits

                def pen_seq(o_flat, a_flat):
                    if pen_name == "L2":
                        return torch.norm(a_flat - o_flat, 2)**2
                    if pen_name == "Cosine":
                        return 1 - F.cosine_similarity(a_flat, o_flat)
                    if pen_name == "KL":
                        o_seq = o_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                        a_seq = a_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                        _, (h_o, _) = info["lstm_layer"](o_seq)
                        h_cat_orig = torch.cat((h_o[-2], h_o[-1]), 1)
                        _, (h_a, _) = info["lstm_layer"](a_seq)
                        h_cat_adv  = torch.cat((h_a[-2], h_a[-1]), 1)
                        logits_o = info["fc_layer"](h_cat_orig)
                        logits_a = info["fc_layer"](h_cat_adv)
                        with torch.no_grad():
                            p0 = F.softmax(logits_o, -1)
                        pa_log = F.log_softmax(logits_a, -1)
                        return F.kl_div(pa_log, p0, reduction="batchmean")
                    return torch.tensor(0., device=device)

                if atk == "Rotation":
                    atk_mod = RotationAttack(flat_o.size(1)).to(device)
                    optim = torch.optim.Adam(atk_mod.parameters(),
                                             lr=args.rotation_lr)
                    x_o = flat_o.clone().detach().requires_grad_(True)
                    for _ in range(args.rotation_steps):
                        adv_flat = atk_mod(x_o)
                        clf, _ = loss_seq(flat_o, adv_flat, y)
                        penv = lam * pen_seq(flat_o, adv_flat) if pen_name else 0.
                        (clf + penv).backward()
                        optim.step(); optim.zero_grad()
                        x_o = adv_flat.detach().requires_grad_(True)
                    adv_flat = x_o

                elif atk == "PGD":
                    adv_flat = PGDAttack(args.pgd_alpha, args.pgd_steps, lam)(
                               flat_o, loss_seq, pen_seq, y)
                else:
                    x_flat = flat_o.clone().detach().requires_grad_(True)
                    for _ in range(args.deepfool_steps):
                        seq = x_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                        _, (h_a, _) = info["lstm_layer"](seq)
                        h_cat3 = torch.cat((h_a[-2], h_a[-1]), 1)
                        logits = info["fc_layer"](h_cat3)
                        o_lbl = logits.argmax(-1)
                        vals, idxs = torch.sort(logits, descending=True)
                        sec = idxs[0,1]
                        info["lstm_layer"].zero_grad(); info["fc_layer"].zero_grad()
                        diff = logits[0, o_lbl] - logits[0, sec]
                        diff.backward()
                        w = x_flat.grad.data
                        r = (abs(diff.item()) + 1e-4) * w / (w.norm()**2)
                        x_flat = (x_flat + (1+0.02)*r).detach().requires_grad_(True)
                        seq2 = x_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                        _, (h2, _) = info["lstm_layer"](seq2)
                        h_cat4 = torch.cat((h2[-2], h2[-1]), 1)
                        if info["fc_layer"](h_cat4).argmax(-1) != o_lbl:
                            break
                    adv_flat = x_flat

                a_seq = adv_flat.view(1, seq_len, emb_dim).permute(1,0,2)
                _, (h_n, _) = info["lstm_layer"](a_seq)
                h_catn = torch.cat((h_n[-2], h_n[-1]), 1)
                if info["fc_layer"](h_catn).argmax(-1).item() == 1 - lbl:
                    success += 1

        asr = 100 * success / total if total else 0
        print(f"{m_name:12s} | {atk:8s} | {loss_name:7s} | "
              f"{pen_name or '-':8s} | {lam:<10g} | {asr:7.2f}")

if __name__ == "__main__":
    main()

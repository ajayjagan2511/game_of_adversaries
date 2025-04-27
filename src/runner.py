#!/usr/bin/env python3
import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# Silence CUDA/XLA alerts
os.environ["TF_CPP_MIN_LOG_LEVEL"]           = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"

# Project imports
from .config    import get_args
from .models    import load_model
from .attacks   import RotationAttack, PGDAttack, DeepFoolAttack
from .losses    import loss_map
from .penalties import pen_map
from .judge     import sem_score                        


curr_model = None
neg_means  = {}
main_mod   = sys.modules["__main__"]
main_mod.curr_model = curr_model
main_mod.neg_means  = neg_means

def main():
    global curr_model, neg_means, main_mod

    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) (model,attack,loss,penalty) combos
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

    # 2) victim models
    victims = {m: load_model(m, device) for m in args.models}

    # 3) LSTM internals
    for info in victims.values():
        if info["type"] == "ta":
            mdl = info["model"]
            info["emb_layer"]  = next(x for x in mdl.modules() if isinstance(x, nn.Embedding))
            info["lstm_layer"] = next(x for x in mdl.modules() if isinstance(x, nn.LSTM))
            info["fc_layer"]   = next(x for x in mdl.modules() if isinstance(x, nn.Linear))

    # 3.5) pre-compute mean negatives for Triplet loss
    train_ds = load_dataset("glue", "sst2", split="train")
    for info in victims.values():
        model = info["model"]
        per = {0: [], 1: []}
        if info["type"] == "hf":
            for ex in train_ds:
                ids = info["tokenizer"](ex["sentence"], return_tensors="pt",
                                         truncation=True, padding=True).to(device)
                with torch.no_grad():
                    emb = info["encoder"](**ids).last_hidden_state[:, 0, :].detach()
                per[ex["label"]].append(emb.squeeze(0))
        else:
            for ex in train_ds:
                tid = torch.tensor(info["tokenizer"](ex["sentence"]),
                                   device=device).unsqueeze(0)
                with torch.no_grad():
                    seq = info["emb_layer"](tid.t())
                    _, (h, _) = info["lstm_layer"](seq)
                    per[ex["label"]].append(torch.cat((h[-2], h[-1]), 1).squeeze(0))
        neg_means[model] = {k: torch.stack(v).mean(0, True) for k, v in per.items()}
    main_mod.neg_means = neg_means  

    dataset = load_dataset("glue", "sst2",
                           split=f"validation[:{args.dataset_size}]")

    # 4) ablation
    for (m_name, atk, loss_name, pen_name), lam in zip(combos, args.lambdas):
        info       = victims[m_name]
        curr_model = info["model"]
        main_mod.curr_model = curr_model    

        success = total = 0
        sem_scores = []                        

        curr_model.train() if info["type"] == "ta" else curr_model.eval()

        for ex in dataset:
            txt, lbl = ex["sentence"], ex["label"]
            emb = None  

       
            if info["type"] == "hf":
                with torch.no_grad():
                    ids  = info["tokenizer"](txt, return_tensors="pt").to(device)
                    emb  = info["encoder"](**ids).last_hidden_state[:, 0, :].detach()
                    pred = curr_model.classifier(emb).argmax(-1).item()
            else:
                ids = torch.tensor(info["tokenizer"](txt), device=device).unsqueeze(0)
                with torch.no_grad():
                    seq = info["emb_layer"](ids.t())
                    _, (h, _) = info["lstm_layer"](seq)
                    h_cat = torch.cat((h[-2], h[-1]), 1)
                    pred  = info["fc_layer"](h_cat).argmax(-1).item()

            if pred != lbl:
                continue
            total += 1
            y = torch.tensor([1 - lbl], device=device)

            # ========== DISTILBERT branch ==========
            if emb is not None:
                if atk == "Rotation":
                    mod = RotationAttack(emb.size(-1)).to(device)
                    opt = torch.optim.Adam(mod.parameters(), lr=args.rotation_lr)
                    for _ in range(args.rotation_steps):
                        adv = mod(emb)
                        clf,_ = loss_map[loss_name](emb, adv, y)
                        pen  = 0. if pen_name is None else lam * pen_map[pen_name](emb, adv)
                        beta = -1 if loss_name == "Triplet" else 1
                        (beta * clf + pen).backward(); opt.step(); opt.zero_grad()
                elif atk == "PGD":
                    pen_fn = (lambda o,a: 0.) if pen_name is None else pen_map[pen_name]
                    adv = PGDAttack(args.pgd_alpha, args.pgd_steps, lam)(
                            emb, loss_map[loss_name], pen_fn, y)
                else:
                    adv = DeepFoolAttack(args.deepfool_steps)(emb)

                # evaluate
                if curr_model.classifier(adv).argmax(-1).item() == 1 - lbl:
                    success += 1
                    sem_scores.append(
                        sem_score(emb.squeeze(0).cpu(), adv.squeeze(0).cpu())
                    )

            # ========== LSTM branch ==========
            else:
                seq_len, _, d = seq.shape
                flat_o = seq.permute(1, 0, 2).reshape(1, -1)

                def loss_seq(o_flat, a_flat, yv):
                  o_seq = o_flat.view(1, seq_len, d).permute(1, 0, 2)
                  _, (h_o, _) = info["lstm_layer"](o_seq)
                  h_cat_orig = torch.cat((h_o[-2], h_o[-1]), 1)     

                  a_seq = a_flat.view(1, seq_len, d).permute(1, 0, 2)
                  _, (h_a, _) = info["lstm_layer"](a_seq)
                  h_cat_adv  = torch.cat((h_a[-2], h_a[-1]), 1)     
                  logits = info["fc_layer"](h_cat_adv)
                  if loss_name == "CE":
                      return F.cross_entropy(logits, yv), logits

                  neg = neg_means[curr_model][yv.item()].to(h_cat_adv.device)  
                  from .losses.triplet import _triplet
                  return _triplet(h_cat_adv, neg, h_cat_orig), logits


                def pen_seq(o_flat, a_flat):
                  if pen_name == "L2":
                      return torch.norm(a_flat - o_flat, 2)**2
                  if pen_name == "Cosine":
                      return 1 - F.cosine_similarity(a_flat, o_flat)
                  if pen_name == "KL":
                      o_seq = o_flat.view(1, seq_len, d).permute(1, 0, 2)
                      a_seq = a_flat.view(1, seq_len, d).permute(1, 0, 2)
               
                      _, (h_o, _) = info["lstm_layer"](o_seq)
                      h_cat_orig = torch.cat((h_o[-2], h_o[-1]), 1)  
                      _, (h_a, _) = info["lstm_layer"](a_seq)
                      h_cat_adv  = torch.cat((h_a[-2], h_a[-1]), 1)  
               
                      logits_o = info["fc_layer"](h_cat_orig)
                      logits_a = info["fc_layer"](h_cat_adv)
                      with torch.no_grad():
                          p0 = F.softmax(logits_o, dim=-1)
                      pa_log = F.log_softmax(logits_a, dim=-1)
                      return F.kl_div(pa_log, p0, reduction="batchmean")
                  return torch.tensor(0., device=device)


                if atk == "Rotation":
                    mod = RotationAttack(flat_o.size(1)).to(device)
                    opt = torch.optim.Adam(mod.parameters(), lr=args.rotation_lr)
                    x = flat_o.detach().requires_grad_(True)
                    for _ in range(args.rotation_steps):
                        adv_flat = mod(x)
                        clf,_ = loss_seq(flat_o, adv_flat, y)
                        pen = lam * pen_seq(flat_o, adv_flat) if pen_name else 0.
                        (clf + pen).backward(); opt.step(); opt.zero_grad()
                        x = adv_flat.detach().requires_grad_(True)
                    adv_flat = x
                elif atk == "PGD":
                    adv_flat = PGDAttack(args.pgd_alpha, args.pgd_steps, lam)(
                               flat_o, loss_seq, pen_seq, y)
                else:
                    x_flat = flat_o.clone().detach().requires_grad_(True)
                    for _ in range(args.deepfool_steps):
                        seq_i = x_flat.view(1, seq_len, d).permute(1,0,2)
                        out, (h_a, _) = info["lstm_layer"](seq_i)
                        h_cat3 = torch.cat((h_a[-2], h_a[-1]), dim=1)
                        logits = info["fc_layer"](h_cat3)
                        o_lbl  = logits.argmax(dim=-1)
                        vals, idxs = torch.sort(logits, descending=True)
                        sec = idxs[0,1]
                        info["lstm_layer"].zero_grad(); info["fc_layer"].zero_grad()
                        diff = logits[0, o_lbl] - logits[0, sec]
                        diff.backward()
                        w = x_flat.grad.data
                        r = (abs(diff.item()) + 1e-4) * w / (w.norm()**2)
                        x_flat = (x_flat + (1+0.02) * r).detach().requires_grad_(True)
                        seq2 = x_flat.view(1, seq_len, d).permute(1,0,2)
                        out2, (h2, _) = info["lstm_layer"](seq2)
                        h_cat4 = torch.cat((h2[-2], h2[-1]), dim=1)
                        if info["fc_layer"](h_cat4).argmax(dim=-1) != o_lbl:
                            break
                    adv_flat = x_flat

                # evaluate
                a_seq = adv_flat.view(1, seq_len, d).permute(1, 0, 2)
                _, (h2, _) = info["lstm_layer"](a_seq)
                if info["fc_layer"](torch.cat((h2[-2], h2[-1]), 1)).argmax(-1).item() == 1 - lbl:
                    success += 1
                    sem_scores.append(
                        sem_score(flat_o.squeeze(0).cpu(), adv_flat.squeeze(0).cpu())
                    )

        asr = 100 * success / total if total else 0
        sem_avg = sum(sem_scores) / len(sem_scores) if sem_scores else 0
        print(f"{m_name:12s} | {atk:8s} | {loss_name:7s} | "
              f"{pen_name or '-':8s} | {lam:<10g} | {asr:7.2f} | {sem_avg:7.2f}")

if __name__ == "__main__":
    main()

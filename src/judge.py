
import random, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load once
_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
_J_TOK    = AutoTokenizer.from_pretrained("gpt2")
_J_MODEL  = AutoModelForCausalLM.from_pretrained("gpt2").to(_DEVICE).eval()

def sem_score(orig_vec: torch.Tensor, adv_vec: torch.Tensor) -> float:

    def vec_to_str(v):
        return " ".join(f"{x:.4f}" for x in v.flatten().tolist()[:32])

    prompt = (
        f"Embedding A:\n{vec_to_str(orig_vec)}\n"
        f"Embedding B:\n{vec_to_str(adv_vec)}\n"
        f"On a scale from 0% (meanings completely different) to 100% "
        "rate how semantically similar the two embeddings are. "
        "Just respond with an integer percentage."
    )

    inp = _J_TOK(prompt, return_tensors="pt").to(_DEVICE)
    gen_ids = _J_MODEL.generate(
        **inp,
        max_new_tokens=4,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        pad_token_id=_J_TOK.eos_token_id,
    )
    new_ids = gen_ids[0][inp["input_ids"].shape[1]:]
    reply   = _J_TOK.decode(new_ids, skip_special_tokens=True)
    m = re.search(r"\d{2,3}", reply)
    score = int(m.group()) if m else lower
    return float(max(lower, min(score, 100)))

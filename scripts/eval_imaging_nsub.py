"""Imaging designs trained at one discretisation, evaluated at many (Supplement Table S3, lower part).

    python scripts/eval_imaging_nsub.py --ckpt "label=path.pt" ["label=path.pt" ...] --out results/tableS_imaging_nsub_eval.md

Each checkpoint (written by run_imaging_sweep.py) is evaluated on the test split of its dataset with
N_sub = own, 15, 18, 24, 30, 60, 120, 240, 480 slices (same anchoring, height scale and shift).  This separates
the height-quantisation error of the N_sub-level soft staircase (a design trained with few slices scores best at
its own N_sub and drops as soon as N_sub changes) from the split-step propagation error.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from dbpm_d2nn.model import load_checkpoint
from dbpm_d2nn.data import imaging_datasets, make_loaders
from dbpm_d2nn.evaluate import evaluate_imaging

p = argparse.ArgumentParser()
p.add_argument("--ckpt", nargs="+", required=True, help='"label=path" pairs')
p.add_argument("--dataset", default="fmnist"); p.add_argument("--grid", type=int, default=200); p.add_argument("--img", type=int, default=128)
p.add_argument("--n_subs", default="15,18,24,30,60,120,240,480")
p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0")
args = p.parse_args()

dev = args.device
train, val, test = imaging_datasets(args.dataset, args.grid, args.img, n_train=1500, n_val=200, n_test=300, split_seed=0)
_, _, te = make_loaders(train, val, test, batch_size=4, seed=0, num_workers=2, test_batch=50)
n_subs = [int(v) for v in args.n_subs.split(",")]
rows = [f"Imaging ({args.dataset}) dBPM designs evaluated at different N_sub (SSIM on the test split; the design's own N_sub in bold)", "",
        "| design | own N_sub | " + " | ".join(f"N_sub={ns}" for ns in n_subs) + " |", "|---|---|" + "---|" * len(n_subs)]
out = {}
for spec in args.ckpt:
    label, path = spec.rsplit("=", 1)
    m = load_checkpoint(path, dev)
    own = m.n_sub
    vals = {}
    for ns in sorted(set(n_subs) | {own}):
        vals[ns] = evaluate_imaging(m.clone(n_sub=ns).to(dev), te, dev, crop=args.img)["ssim"]
    out[label] = {"own_n_sub": own, "ssim": vals, "path": path}
    cells = [f"**{vals[ns]:.3f}**" if ns == own else f"{vals[ns]:.3f}" for ns in n_subs]
    rows.append(f"| {label} | {own} | " + " | ".join(cells) + " |")
    print(rows[-1], flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
open(args.out, "w").write("\n".join(rows) + "\n")
json.dump(out, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
print("wrote", args.out)

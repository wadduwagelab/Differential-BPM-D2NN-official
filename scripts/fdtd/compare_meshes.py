import json, numpy as np, sys
run_dir, export = sys.argv[1], sys.argv[2]
res = json.load(open(f"{run_dir}/results.json"))
d = np.load(export)
def nmse_c(a, b):
    a=a.ravel(); b=b.ravel(); al = np.vdot(a, b)/np.vdot(a, a); return float(np.sum(np.abs(al*a-b)**2)/np.sum(np.abs(b)**2))
def rel_l2(a, b):
    a=a.ravel(); b=b.ravel(); al=float(np.dot(a,b)/np.dot(a,a)); return float(np.linalg.norm(al*a-b)/np.linalg.norm(b))
patches = json.load(open(export.replace("fdtd_export.npz","summary.json")))["meta"]["patches"]
labels = d["labels"]
def pred(I):
    e=[I[y0:y1,x0:x1].sum() for (y0,y1,x0,x1) in patches]; return int(np.argmax(e))
for s in (0,1):
    for des in ("thin","bpm"):
        r10 = np.load(res[f"s{s:02d}_{des}_mesh10"]["file"]); r20 = np.load(res[f"s{s:02d}_{des}_mesh20"]["file"])
        I10, I20 = r10["I"], r20["I"]; E10, E20 = r10["Ey"], r20["Ey"]
        Ith = d[f"I_{des}design_thinmodel"][s]; Ibp = d[f"I_{des}design_bpm_finemodel"][s]
        Uth = d[f"U_{des}design_thinmodel"][s]; Ubp = d[f"U_{des}design_bpm_finemodel"][s]
        print(f"s{s} {des}: label {labels[s]} | pred mesh10 {pred(I10)} mesh20 {pred(I20)} thin-model {pred(Ith)} bpm-model {pred(Ibp)}")
        print(f"   FDTD mesh10 vs mesh20: rel-L2 intensity {rel_l2(I10,I20):.3f}, field NMSE {nmse_c(E10,E20):.4f}")
        print(f"   thin-model vs FDTD: mesh10 {nmse_c(Uth,E10):.3f}  mesh20 {nmse_c(Uth,E20):.3f} | bpm-model vs FDTD: mesh10 {nmse_c(Ubp,E10):.3f}  mesh20 {nmse_c(Ubp,E20):.3f}")
        print(f"   costs: mesh10 {res[f's{s:02d}_{des}_mesh10'].get('real_cost')}, mesh20 {res[f's{s:02d}_{des}_mesh20'].get('real_cost')}")

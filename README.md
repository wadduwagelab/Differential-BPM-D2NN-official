# dbpm_d2nn — differentiable beam-propagation (∂BPM) layers for D²NNs

Training code for finite-thickness diffractive layers. Each layer can be a thin
phase mask or a volumetric ∂BPM stack. The same height maps can be checked with
Tidy3D FDTD.

Paper: [Jayakody & Wadduwage, arXiv:2606.07896](https://arxiv.org/abs/2606.07896).

```
.
├── dbpm_d2nn/                 # importable package
│   ├── propagation.py         # angular-spectrum propagator
│   ├── model.py               # D2NN: thin or volumetric (∂BPM) layers
│   ├── data.py                # MNIST / Fashion-MNIST / CIFAR-100 loaders
│   ├── train.py               # classification and imaging training
│   ├── evaluate.py            # accuracy, imaging metrics, field NMSE
│   ├── calibrate.py           # global axial offset δz and height scale s
│   ├── metrics.py             # SSIM, NMSE, Pearson r
│   ├── stats.py               # McNemar, bootstrap CIs, paired tests
│   └── fdtd.py                # Tidy3D geometry, source, monitors, binning
├── scripts/
│   ├── run_classification_sweep.py   # classification, 3 seeds by default
│   ├── run_imaging_sweep.py          # imaging (same geometry)
│   └── fdtd/
│       ├── train_fdtd_designs.py     # train 32×32 designs and export samples
│       ├── run_single_layer.py       # one layer through thin / ∂BPM / FDTD
│       ├── run_multilayer.py         # paired 5-layer FDTD
│       ├── compare_meshes.py         # λ/10 vs λ/20 check
│       └── analyze_fdtd.py           # task metrics and model-vs-FDTD fidelity
├── configs/
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch with CUDA is strongly recommended
export PYTHONPATH=$PWD                    # scripts import dbpm_d2nn from the repo root
```

Datasets are downloaded by `torchvision` on first use (`DBPM_DATA_ROOT` changes the
location). FDTD needs a Flexcompute account (`tidy3d configure` after
`pip install tidy3d`). Everything else runs offline.

## The ∂BPM layer

```python
from dbpm_d2nn.model import D2NN
lam, dx = 532e-9, 0.535 * 532e-9
thin = D2NN(H=200, W=200, L=5, wavelength=lam, dx=dx, z=40 * lam, n_material=1.5, n_sub=1)
vol  = D2NN(H=200, W=200, L=5, wavelength=lam, dx=dx, z=40 * lam, n_material=1.5, n_sub=24,
            axial_mode="anchor", anchor=0.5)   # relief centroid on the nominal plane
vol.load_layers_from(thin)
I = vol(u_in)                                  # detector intensity; differentiable in the phases
```

`n_sub` is the number of slices per \(h_{\max}\). Pass `dz=` instead to fix the
slice thickness. `axial_mode="legacy"` puts the relief entrance on the nominal
plane.

## Train ∂BPM (3 seeds)

Classification defaults to seeds `0,1,2`. One call loops over refractive index
and seeds and writes `results.json`.

```bash
python scripts/run_classification_sweep.py --dataset mnist  --encoding phase     --out_dir experiments/cls/mnist_phase
python scripts/run_classification_sweep.py --dataset mnist  --encoding amplitude --out_dir experiments/cls/mnist_amp
python scripts/run_classification_sweep.py --dataset fmnist --encoding phase     --out_dir experiments/cls/fmnist_phase
python scripts/run_classification_sweep.py --dataset fmnist --encoding amplitude --out_dir experiments/cls/fmnist_amp
```

Imaging (same 200×200, \(0.535\lambda\) pitch, \(40\lambda\) spacing):

```bash
python scripts/run_imaging_sweep.py --dataset mnist --out_dir experiments/img/mnist
python scripts/run_imaging_sweep.py --dataset fmnist --out_dir experiments/img/fmnist
python scripts/run_imaging_sweep.py --dataset cifar100 --out_dir experiments/img/cifar100
```

Geometry and defaults are in `configs/classification.yaml` and `configs/imaging.yaml`.

## FDTD

Train small 32×32 designs, export a stratified test set, run Tidy3D, then score
the detector fields.

```bash
pip install tidy3d
tidy3d configure

python scripts/fdtd/train_fdtd_designs.py --task classification --n 1.5 --z_lam 8 \
    --out_dir experiments/fdtd/designs_cls

python scripts/fdtd/run_single_layer.py \
    --export experiments/fdtd/designs_cls/fdtd_export.npz \
    --out_dir experiments/fdtd/single_layer

python scripts/fdtd/run_multilayer.py \
    --export experiments/fdtd/designs_cls/fdtd_export.npz \
    --out_dir experiments/fdtd/multilayer_cls_n1.5 --mesh 10

python scripts/fdtd/analyze_fdtd.py \
    --run_dir experiments/fdtd/multilayer_cls_n1.5 \
    --export experiments/fdtd/designs_cls/fdtd_export.npz \
    --task classification --mesh 10
```

Imaging FDTD is the same with `--task imaging`. See `configs/fdtd.yaml` for
geometry and mesh settings.

## Citation

```bibtex
@article{jayakody2026beyond,
  title   = {Beyond the Thin-Layer Limit: Differentiable Volumetric Training
             for Visible-Range Diffractive Neural Networks},
  author  = {Jayakody, Dineth and Wadduwage, Dushan N.},
  journal = {arXiv preprint arXiv:2606.07896},
  year    = {2026}
}
```

## License

MIT (see LICENSE).

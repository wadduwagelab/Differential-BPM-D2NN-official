"""
dbpm_d2nn - Differentiable beam-propagation (dBPM) layers for diffractive deep
neural networks (D2NNs).

* ``propagation``  - angular-spectrum (ASM) free-space propagator with cached
                     transfer functions,
* ``model``        - the D2NN forward model.  Every diffractive layer can be
                     evaluated as a thin phase mask or as a finite-thickness
                     volume (dBPM layer).  The volume can be anchored at its
                     entrance, centroid, or an arbitrary fraction of its
                     thickness; a global axial shift and a global height-scale
                     factor are optional,
* ``data``         - MNIST, Fashion-MNIST, and CIFAR-100 loaders for
                     classification and imaging,
* ``train``        - training / fine-tuning loops,
* ``evaluate``     - accuracy, imaging metrics, field NMSE,
* ``calibrate``    - one-parameter post-hoc calibration (global axial offset,
                     global height scale),
* ``metrics``      - SSIM, NMSE, Pearson r, fabrication-geometry statistics,
* ``stats``        - exact McNemar test, Wilcoxon, paired bootstrap CIs,
* ``fdtd``         - Tidy3D geometry, sources, batch runs, and post-processing.
"""

__version__ = "1.0.0"

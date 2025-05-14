# [SmoothHess: ReLU Network Feature Interactions via Stein's Lemma (NeurIPS 2023)](https://openreview.net/pdf?id=dwIeEhbaD0)

[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/9ef5e965720193681fc8d16372ac4717-Paper-Conference.pdf) [Poster](https://neurips.cc/virtual/2023/poster/70998) [Video](https://neurips.cc/virtual/2023/poster/70998) 

[Max Torop,*](https://maxtorop.github.io/) [Aria Masoomi,*](https://scholar.google.com/citations?user=KXcX8coAAAAJ&hl=en) [Davin Hill,](https://www.davinhill.me/) [Kivanc Kose,](https://kkose.github.io/about/) [Stratis Ioannidis](https://ece.northeastern.edu/fac-ece/ioannidis/) and [Jennifer Dy](https://mllabneu.github.io/)




### 📘 Abstract

> Several recent methods for interpretability model feature interactions by looking at the Hessian of a neural network. This poses a challenge for ReLU networks, which are piecewise-linear and thus have a zero Hessian almost everywhere. We propose SmoothHess, a method of estimating second-order interactions through Stein's Lemma. In particular, we estimate the Hessian of the network convolved with a Gaussian through an efficient sampling algorithm, requiring only network gradient calls. SmoothHess is applied post-hoc, requires no modifications to the ReLU network architecture, and the extent of smoothing can be controlled explicitly. We provide a non-asymptotic bound on the sample complexity of our estimation procedure. We validate the superior ability of SmoothHess to capture interactions on benchmark datasets and a real-world medical spirometry dataset.


### Code written in Python 3.8.3 using the following packages
- torch 1.13.1+cu117
- torchvision 0.11.1
- matplotlib 3.7.1
- numpy 1.25.5
- pandas 1.5.3
- cvxpy 1.2.1
- tqdm 4.55.0

### Overview

- [`FourQuadrant.ipynb`](./FourQuadrant.ipynb): Demonstrates the Four Quadrant dataset experiment, highlighting the intuitive control offered by **SmoothHess** compared to the SoftPlus Hessian.
- [`PMSEExample.ipynb`](./PMSEExample.ipynb): Example notebook for the **Perturbation Mean-Squared Error** (PMSE) experiment, demonstrating the strong ability of SmoothHess to capture the networks local behaviour.
- [`AdvAttackExample.ipynb`](./AdvAttackExample.ipynb): Example notebook demonstrating how to perform adversarial attacks using SmoothHess.


### Citation
If you use this code, please cite our paper:

```bibtex
@article{torop2023smoothhess,
  title={SmoothHess: ReLU network feature interactions via stein's lemma},
  author={Torop, Max and Masoomi, Aria and Hill, Davin and Kose, Kivanc and Ioannidis, Stratis and Dy, Jennifer},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={50697--50729},
  year={2023}
}

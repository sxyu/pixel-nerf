# pixelNeRF: Neural Radiance Fields from One or Few Images

Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa<br>
UC Berkeley

**Note: We are in the process of cleaning up code and porting experiments to this repository.
Currently, an implementation of the core model is provided.**
Some untested changes were made after refactoring.
Complete training code and more documentation are coming soon, please stay tuned.

![Teaser](https://raw.github.com/sxyu/pixel-nerf/master/readme-img/paper_teaser.jpg)

arXiv: http://arxiv.org/abs/2012.02190

This is the official code repository of our paper, pixelNeRF.

# Datasets

- DTU dataset. Please obtain from 
https://drive.google.com/drive/folders/195u2QBAB3apozHZNVOcGuKSognKlpw0A?usp=sharing

- ShapeNet 64x64 dataset from NMR, for main ShapeNet experiments. Obtain from
https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
(Hosted by DVR authors) 

- SRN experiments.
  - SRN *fixed* car dataset from
    https://drive.google.com/file/d/1AUzjr6_JGXvZ1N2eqjOol2wwEZ8E2Pk_/view?usp=sharing
  - SRN chair dataset from (TBA)

- Two-object experiments.
Obtain our dataset from
https://drive.google.com/drive/folders/195u2QBAB3apozHZNVOcGuKSognKlpw0A?usp=sharing

While we could have used a common data format, we chose to keep
DTU and ShapeNet (NMR) datasets in DVR's format and SRN data in original SRN format.
Our own two-object data is in NeRF's format.
Data adapters are built into the code.

# Instructions

TBA

# Training instructions

TBA

# BibTeX

```
@misc{yu2020pixelnerf,
      title={pixelNeRF: Neural Radiance Fields from One or Few Images}, 
      author={Alex Yu and Vickie Ye and Matthew Tancik and Angjoo Kanazawa},
      year={2020},
      eprint={2012.02190},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgements

Parts of the code were based on from kwea123's NeRF implementation: https://github.com/kwea123/nerf_pl.
Some functions are borrowed from DVR https://github.com/autonomousvision/differentiable_volumetric_rendering
and PIFu https://github.com/shunsukesaito/PIFu

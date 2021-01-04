# pixelNeRF: Neural Radiance Fields from One or Few Images

Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa<br>
UC Berkeley

![Teaser](https://raw.github.com/sxyu/pixel-nerf/master/readme-img/paper_teaser.jpg)

arXiv: http://arxiv.org/abs/2012.02190

This is a *temporary* code repository for our paper, pixelNeRF, pending final release.

# Datasets

- For the main ShapeNet experiments, we use the ShapeNet 64x64 dataset from NMR
https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
(Hosted by DVR authors) 
    - For novel-category generalization experiment, a custom split is needed.
      Download the following script:
      https://drive.google.com/file/d/1Uxf0GguAUTSFIDD_7zuPbxk1C9WgXjce/view?usp=sharing
      place the said file under `NMR_Dataset` and run `python genlist.py` in the said directory.
      This generates train/val/test lists for the experiment. Note for evaluation performance reasons,
      test is only 1/4 of the unseen categories.

- The remaining datasets may be found in
https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing
    - Custom two-chair `multi_chair_*.zip`
    - DTU (4x downsampled, rescaled) in DVR's DTU format `dtu_dataset.zip`
    - SRN chair/car (128x128) `srn_*.zip`
      note the car set is a re-rendered version provided by Vincent Sitzmann

While we could have used a common data format, we chose to keep
DTU and ShapeNet (NMR) datasets in DVR's format and SRN data in the original SRN format.
Our own two-object data is in NeRF's format.
Data adapters are built into the code (via the flag `-F <multi_obj | dvr | dvr_gen | dvr_dtu | srn>`).

# Running the model (generate video)

## DTU

1. Download DTU dataset (see Datasets section). Extract to some directory, to get: `<data_root>/rs_dtu_4`
2. Download the pretrained DTU model
3. Run using `python gen_video.py  -n dtu -c conf/dtu.conf --gpu_id=<GPU> --split val -P '22 25 28'  -D <data_root>/rs_dtu_4 -F dvr_dtu -S 3  --ray_batch_size=20000 --black --scale 0.25`

Replace `<GPU>` with desired GPU id. Replace `-S 3` with `-S <number>` to run on a different scene.
Remove `--scale 0.25` to render at full reso (slow)

Note that for this dataset I only use train/val sets, where val is used for test.

## ShapeNet 64x64

1. Download NMR ShapeNet renderings (see Datasets section, 1st link)
2. Download the pretrained shapenet category-agnostic model
3. Run using `python gen_video.py  -n sn64 -c conf/default_mv.conf --gpu_id=<GPU> --split test -P '22 25 28'  -D <data_root>/rs_dtu_4 -F dvr -S 2 --ray_batch_size=20000`

Replace `<GPU>` with desired GPU id.  Replace `-S 3` with `-S <number>` to run on a different object.

## Real Car Images

**Note: requires PointRend from detectron2.**
Install detectron2 by following https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md.

1. Download any car image.
Place it in `<project dir>/input`. Some example images are shipped with the repo.
The car should be fully visible.
2. Download the pretrained *SRN car* model.
3. Run the preprocessor script: `python scripts/preproc.py`. This saves `input/*_normalize.png`.
If the result is not reasonable, PointRend didn't work; please try another imge.
4. Run `python eval_real.py`. Outputs will be in `<project dir>/output`

The Stanford Car dataset contains many example car images:
<https://ai.stanford.edu/~jkrause/cars/car_dataset.html>.
Note the normalization heuristic has been slightly modified compared to the paper. There may be some minor differences.

# Quantitative evaluation instructions

`eval.py`

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

# pixelNeRF: Neural Radiance Fields from One or Few Images

Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa<br>
UC Berkeley

![Teaser](https://raw.github.com/sxyu/pixel-nerf/master/readme-img/paper_teaser.jpg)

arXiv: http://arxiv.org/abs/2012.02190

This is a *temporary* code repository for our paper, pixelNeRF, pending final release.
The official repository shall be <https://github.com/sxyu/pixel-nerf>.

# Getting the data

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
Data adapters are built into the code.

# Running the model (video generation)

The main implementation is in the `src/` directory,
while evalutation scripts are in `eval/`.

Download all pretrained weight files from
<https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju/view?usp=sharing>.
Extract this to `<project dir>/checkpoints/`, so that `<project dir>/checkpoints/dtu/pixel_nerf_latest` exists.


## ShapeNet 64x64

1. Download NMR ShapeNet renderings (see Datasets section, 1st link)
2. Download the pretrained shapenet 64 models
3. Run using 
    - `python eval/gen_video.py  -n sn64 -c conf/sn64.conf --gpu_id <GPU> --split test -P '2'  -D <data_root>/NMR_Dataset -F dvr -S 0 --ray_batch_size=20000`
    - For unseen category generalization:
      `python eval/gen_video.py  -n sn64_unseen -c conf/sn64.conf --gpu_id=<GPU> --split test -P '2'  -D <data_root>/NMR_Dataset -F dvr_gen -S 0 --ray_batch_size=20000`

Replace `<GPU>` with desired GPU id.  Replace `-S 0` with `-S <number>` to run on a different ShapeNet object id.
Replace `-P '2'` with `-P '<number>'` to use a different input view.
Replace `--split test` with `--split train | val` to use different split.
Adjust `--ray_batch_size` if running out of memory.

Pre-generated results for all ShapeNet objects with comparison may be found at <https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/>

## DTU

1. Download DTU dataset (see Datasets section). Extract to some directory, to get: `<data_root>/rs_dtu_4`
2. Download the pretrained DTU model
3. Run using `python eval/gen_video.py  -n dtu -c conf/dtu.conf --gpu_id=<GPU> --split val -P '22 25 28'  -D <data_root>/rs_dtu_4 -F dvr_dtu -S 3  --ray_batch_size=20000 --black --scale 0.25`

Replace `<GPU>` with desired GPU id. Replace `-S 3` with `-S <number>` to run on a different scene.
Remove `--scale 0.25` to render at full resolution (quite slow).

Note that for DTU, I only use train/val sets, where val is used for test. This is due to the very small size of the dataset. 
The model overfits to the train set significantly during training.

## Real Car Images

**Note: requires PointRend from detectron2.**
Install detectron2 by following https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md.

1. Download any car image.
Place it in `<project dir>/input`. Some example images are shipped with the repo.
The car should be fully visible.
2. Download the pretrained *SRN car* model.
3. Run the preprocessor script: `python scripts/preproc.py`. This saves `input/*_normalize.png`.
If the result is not reasonable, PointRend didn't work; please try another imge.
4. Run `python eval/eval_real.py`. Outputs will be in `<project dir>/output`

The Stanford Car dataset contains many example car images:
<https://ai.stanford.edu/~jkrause/cars/car_dataset.html>.
Note the normalization heuristic has been slightly modified compared to the paper. There may be some minor differences.
You can pass `-e -20' to `eval_real.py` to set the elevation higher in the generated video.

# Overview of flags

Generally, all scripts in the project take the following flags
- `-c <conf/*.conf>`: config file
- `-n <expname>`: experiment name, matching checkpoint directory name
- `-F <multi_obj | dvr | dvr_gen | dvr_dtu | srn>`: data format
- `-D <datadir>`: data directory. For SRN/multi_obj datasets with 
    separate directories e.g. `path/cars_train`, `path/cars_val`,
    put `-D path/cars`
- `--split <train | val | test>`: data set split
- `-S <subset_id>`: scene or object id to render
- `--gpu_id <GPU>`: GPU id to use
- `--ray_batch_size <sz>`: Batch size of rendered rays. Default is 50000; make it smaller if you run out of memory.
    On large-memory GPUs, set it to 100000 for eval for better performance.

Please refer the the following table

| Name                       | expname -n | config -c            | data format -F | Data file                               | data dir -D         |
|----------------------------|------------|----------------------|----------------|-----------------------------------------|---------------------|
| ShapeNet category-agnostic | sn64       | conf/sn64.conf       | dvr            | NMR_Dataset.zip (from AWS)              | path/NMR_Dataset  |
| ShapeNet unseen category   | sn64_unseen   | conf/sn64.conf       | dvr_gen        | NMR_Dataset.zip (from AWS) + genlist.py | path/NMR_Dataset |
| SRN chairs                 | srn_chair  | conf/default_mv.conf | srn            | srn_chairs.zip                          | path/chairs       |
| SRN cars                   | srn_car    | conf/default_mv.conf | srn            | srn_cars.zip                            | path/cars         |
| DTU                        | dtu        | conf/dtu.conf        | dvr_dtu        | dtu_dataset.zip                         | path/rs_dtu_4     |
| Two chairs                 | TBA        | TBA                  | multi_obj      | multi_chair_*.zip                       | path              |


# Quantitative evaluation instructions

All evaluation code is in `eval/` directory.
The full, parallelized evaluation code is in `eval/eval.py`.

## Approximate Evaluation
The full evaluation can be extremely slow (taking many days).
Therefore we also provide `eval_approx.py` for *approximate* evaluation.

- Example `python eval/eval_approx.py -F srn -D <srn_data>/cars -n srn_car -c conf/default_mv.conf`

Add `--seed <number>` to try a different random seed.

## Full Metric Evaluation

Here we provide commands for full evaluation with `eval.py`.
Append `--gpu_id=<GPU1>` after each command, and add `--extra_gpus=GPU ids separated by space` to use multiple GPUs.
Resume-capability is built-in, and you can simply run the command again to resume if the process is terminated.

In all cases, a source-view specification is required. This can be either `-P` or `-L`. `-P 'view1 view2..'` specifies 
a set of fixed input views. In contrast, `-L` should point to a viewlist file (viewlist/src_*.txt) which specifies views to use for each object.

`-O <dirname>` specifies output directory name. Renderings and progress will be saved to this directory.

### ShapeNet

- Category-agnostic eval `python eval/eval.py -F dvr -D <path>/NMR_Dataset -n sn64 -c conf/sn64.conf -L viewlist/src_dvr.txt --multicat -O eval_sn64`
- Unseen category eval `python eval/eval.py -F dvr_gen -D <path>/NMR_Dataset -n sn64_unseen -c conf/sn64.conf -L viewlist/src_gen.txt --multicat -O eval_sn64_unseen`

### SRN ShapeNet

- SRN car 1-view eval `python eval/eval.py -F srn -D <srn_data>/cars -n srn_car -c conf/default_mv.conf -P '64' -O srn_car_1v`
- SRN car 2-view eval `python eval/eval.py -F srn -D <srn_data>/cars -n srn_car -c conf/default_mv.conf -P '64 104' -O srn_car_2v`

The command for chair is analogous (replace car with chair). The input views 64, 104 are taken from SRN.
Our method is by no means restricted to using such views.

### DTU
- 1-view `python eval/eval.py -F dvr_dtu -D <data>/rs_dtu_4 -n dtu -c conf/dtu.conf -P '25' -O dtu_1v`
- 3-view `python eval/eval.py -F dvr_dtu -D <data>/rs_dtu_4 -n dtu -c conf/dtu.conf -P '22 25 28' -O dtu_3v`
- 6-view `python eval/eval.py -F dvr_dtu -D <data>/rs_dtu_4 -n dtu -c conf/dtu.conf -P '22 25 28 40 44 48' -O dtu_6v`
- 9-view `python eval/eval.py -F dvr_dtu -D <data>/rs_dtu_4 -n dtu -c conf/dtu.conf -P '22 25 28 40 44 48 0 8 13' -O dtu_9v`

In training, we always provide 3-views, so the improvement with more views is limited.

# Training instructions

Training code is in `train/` directory.
Check out `train/train.py`. More information to come.

- Example for training to DTU: `python train/train.py -n dtu_exp -c conf/dtu.conf -D <data dir>/rs_dtu_4 -F dvr_dtu -V 3 --gpu_id=<GPU> --resume`

Additional flags
- `--resume` to resume from checkpoint, if available
- `-V <number>` to specify number of views.
    - `-V 'numbers separated by space'` to use random number of views per batch. This does not work so well in our experience but we use it for SRN experiment.

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

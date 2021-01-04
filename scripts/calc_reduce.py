"""
Collect all metrics data aftef running eval and calc_metrics
"""
import os
import argparse
import tqdm
import json

parser = argparse.ArgumentParser(
    description="Reduce metrics."
)

parser.add_argument(
    "--output", "-O", type=str, default="eval_dvr",
    help="Root path of rendered results from eval and calc_metrics (custom format)"
)
parser.add_argument("--multicat", action="store_true",
        help="Prepend category id to object id. Specify if model fits multiple categories.")
parser.add_argument(
    "--metadata", type=str, default="/home/sxyu/proj/DVR/data/NMR_Dataset/metadata.yaml",
    help="Path to dataset metadata, used if --multicat"
)
parser.add_argument("--dtu_sort", action="store_true",
        help="Sort using DTU scene order instead of lex")
args = parser.parse_args()

render_root = args.output

if args.multicat:
    meta = json.load(open(args.metadata, 'r'))
    cats = sorted(list(meta.keys()))
    cat_description = {cat : meta[cat]['name'].split(',')[0] for cat in cats}

all_objs = []
objs = [x for x in os.listdir(render_root)]
objs = [os.path.join(render_root, x) for x in objs if x[0] != '_']
objs = [x for x in objs if os.path.isdir(x)]
if args.dtu_sort:
    objs = sorted(objs, key=lambda x: int(x[x.rindex('/') + 5:]))
else:
    objs = sorted(objs)
all_objs.extend(objs)

print(">>> PROCESSING", len(all_objs), 'OBJECTS')

METRIC_NAMES = ['psnr', 'ssim', 'lpips']

out_metrics_path = os.path.join(render_root, 'all_metrics.txt')

if args.multicat:
    cat_sz = {}
    for cat in cats:
        cat_sz[cat] = 0

all_metrics = {}
for name in METRIC_NAMES:
    if args.multicat:
        for cat in cats:
            all_metrics[cat + '.' + name] = 0.0
    all_metrics[name] = 0.0

for obj_root in tqdm.tqdm(all_objs):
    metrics_path = os.path.join(obj_root, 'metrics.txt')
    with open(metrics_path, 'r') as f:
        metrics = [line.split() for line in f.readlines()]
    if args.multicat:
        cat_name = os.path.basename(obj_root).split('_')[0]
        cat_sz[cat_name] += 1
        for metric, val in metrics:
            all_metrics[cat_name + '.' + metric] += float(val)
    print(obj_root, end=' ')
    for metric, val in metrics:
        all_metrics[metric] += float(val)
        print(val, end=' ')
    print()

for name in METRIC_NAMES:
    if args.multicat:
        for cat in cats:
            if cat_sz[cat] > 0:
                all_metrics[cat + '.' + name] /= cat_sz[cat]
    all_metrics[name] /= len(all_objs)
    print(name, all_metrics[name])

metrics_txt = []
if args.multicat:
    for cat in cats:
        if cat_sz[cat] > 0:
            cat_txt = '{:12s}'.format(cat_description[cat])
            for name in METRIC_NAMES:
                cat_txt += ' {}: {:.6f}'.format(name, all_metrics[cat + '.' + name])
            cat_txt += ' n_inst: {}'.format(cat_sz[cat])
            metrics_txt.append(cat_txt)

    total_txt = '---\n{:12s}'.format('total')
else:
    total_txt = ''
for name in METRIC_NAMES:
    total_txt += ' {}: {:.6f}'.format(name, all_metrics[name])
metrics_txt.append(total_txt)

metrics_txt = '\n'.join(metrics_txt)
with open(out_metrics_path, 'w') as f:
    f.write(metrics_txt)
print('WROTE', out_metrics_path)
print(metrics_txt)

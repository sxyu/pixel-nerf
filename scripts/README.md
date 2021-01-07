# Rendering multiple object ShapeNet scenes
The `render_shapenet.py` script is used to render ShapeNet scenes composed of multiple object instances, given an object class.
This script will render different splits (train/test/val) of the ShapeNet models; see [Render Flags](#render-flags) for more information.

## Installing Blender

1. Download and untar Blender
```
wget https://mirror.clarkson.edu/blender/release/Blender2.90/blender-2.90.1-linux64.tar.xz
tar -xvf blender-2.82a-linux64.tar.xz 
```

2. Install other Python dependencies in the Blender bundled Python
```
cd $INSTALL_PATH/blender-2.82a-linux64/2.82/python/bin/
./python3.7m -m ensurepip
./pip3 install numpy scipy
```

3. In your `.bash_aliases` file, add
```
alias bpy="blender --background -noaudio --python”
```
This allows you to call
```
bpy render_shapenet.py -- <flags>
```
Unless debugging, recommended to redirect Blender's stdout to /dev/null and direct stderr to stdout to keep script logging.
```
bpy render_shapenet.py -- <flags> 2>&1 >/dev/null
```

## Render Flags
- `--out_dir` (required) -- Parent directory to write rendered images. Instances will be rendered by ID in child subdirectories.
- `--src_model_dir` (required) -- Location of the ShapeNet model directory with all object classes and instances.
- `--object` (default: chair) -- Name of object class to render.
- `--val_frac` (default: 0.2) -- When generating a split of object instances, what fraction of all instances to use as validation. The resulting split is written in the object class directory as `val_split_{n_val}.txt`.
- `--test_frac` (default: 0.2) -- When generating a split of object instances, what fraction of all instances to use as test. The resulting split is written in the object class directory as `test_split_{n_test}.txt`.
- `--split` (choice of `[train, val, test]`) -- Which split to render. `val/test` splits use a specific camera trajectory (Archimedes spiral, from SRN).
- `--n_views` (default: 20) -- Number of views to render per instance.
- `--res` (default: 128) -- Output resolution of images (default 128x128).
- `--start_idx` (default: 0) -- If rendering a subset of the object instances, provide the starting index.
- `--n_objects` (default: 2) -- The number of objects to include per scene.
- `--end_idx` (default: -1) -- If rendering a subset of the object instances, provide the ending index.
- `--use_pbr` -- Whether to use Cycles to render with physically based rendering. Slower, but more photorealistic.
- `--light_env` -- If `--use_pbr`, you can use an HDRI light map. Pass the path of the HDRI here.
- `--light_strength` -- The strength of the light map in the scene, if using an HDRI light map. You can easily get HDRIs from websites like https://hdrihaven.com/.
- `--render_alpha` -- Render the object masks.
- `--render_depth` -- Render the scene depth map.
- `--render_bg` -- Render the scene background (only useful if using PBR + HDRI light maps).
- `--pool` -- Render in parallel. Faster.


## Rendering with Blender EEVEE

By default, it is only possible to render headless using Blender's PBR engine Cycles.
While photorealistic, rendering with Eevee is much faster.
To enable headless rendering using Eevee, you also need the following dependencies

### OpenGL
Openg GL is necessary for Virtual GL. Normally OpenGL can be installed through apt.
```sudo apt-get install freeglut3-dev mesa-utils```

### Virtual GL
Install VGL with [this tutorial](https://virtualgl.org/vgldoc/2_2_1/#hd004001).


### TurboVNC
Install TurboVNC with [this tutorial](https://cdn.rawgit.com/TurboVNC/turbovnc/2.1.1/doc/index.html#hd005001).

### X11 utilities
```
sudo apt install x11-xserver-utils libxrandr-dev
```

### Emulating the Virtual Display
First configure your X server to be compatible with your graphics card.
```
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
```
You can also further edit this configuration at `/etc/X11/xorg.conf`.
Now start your X server, labeled with an arbitrary server number, in this case 7
```
sudo nohup Xorg :7 &
```
Run an auxiliary remote VNC server to create a virtual display. Label it with a separate remote server number, in this case 8.
```
/opt/TurboVNC/bin/vncserver :8
```
To test, run `glxinfo` on Xserver 7, device 0 (GPU 0 on your machine).
```
DISPLAY=:8 vglrun -d :7.0 glxinfo
```
If all is well, proceed to run headless rendering with Eevee with
```
DISPLAY=:8 vglrun -d :7.0 blender --background -noaudio --python render_shapenet.py -- <flags>
```

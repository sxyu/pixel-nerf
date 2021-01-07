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
bpy render_shapenet.py -- [ARGS]
```

## Render Arguments


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
Start your X server, labeled with <server number>
```
sudo nohup Xorg :<server number> &
```
Run an auxiliary remote VNC server to create a virtual display. Label it with a separate <remote server number>.
```
/opt/TurboVNC/bin/vncserver :<remote server number>
```
To test, run `glxinfo` on Xserver <server number>, device <device number> (GPU 0 on your machine).
```
DISPLAY=:8 vglrun -d :<server number>.<device number> glxinfo
```
If all is well, headless rendering with Eevee should now work.



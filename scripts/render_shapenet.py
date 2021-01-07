import argparse
from concurrent.futures import ProcessPoolExecutor
from dotmap import DotMap
import glob
import json
import os
import os.path as osp
import shutil
import sys
from time import time

import bpy
from mathutils import Vector
import numpy as np
from numpy.random import Generator, MT19937, SeedSequence


def print_info(*args):
    """
    Print to stderr (so we can send out the blender output in stdout to /dev/null)
    """
    print("INFO:", *args, file=sys.stderr)


def add_lamps():
    bpy.ops.object.light_add(type="SUN", location=(6, 2, 5))
    lamp = bpy.context.object
    lamp.rotation_euler = (-0.5, 0.5, 0)

    bpy.ops.object.light_add(type="SUN", location=(6, -2, 5))
    lamp = bpy.context.object
    lamp.rotation_euler = (-0.5, -0.5, 0)


def import_object(model_dir, model_path, axis_forward="-Z", axis_up="Y"):
    """Load object and get the vertex bounding box"""
    # Deselect all
    for o in bpy.data.objects:
        o.select_set(False)

    name = osp.basename(model_dir)
    path = osp.join(model_dir, model_path)
    bpy.ops.import_scene.obj(filepath=path, axis_forward=axis_forward, axis_up=axis_up)

    # merge all the meshes into one mesh
    selected_objs = bpy.context.selected_objects
    if len(selected_objs) > 1:
        ctx = bpy.context.copy()
        ctx["active_object"] = selected_objs[0]
        ctx["selected_editable_objects"] = selected_objs
        bpy.ops.object.join(ctx)

    obj = selected_objs[0]
    obj.select_set(state=True)

    # randomly rotate object about Z axis
    obj.rotation_euler[2] = np.random.uniform(0, 2 * np.pi)

    # find the proper scaling
    verts = np.array([vert.co for vert in obj.data.vertices])
    verts_max = np.max(verts, axis=0)
    verts_min = np.min(verts, axis=0)

    bb_max = obj.matrix_world @ Vector(verts_max)
    bb_min = obj.matrix_world @ Vector(verts_min)

    # we want the diameter of the object to be ~2 units
    scale = np.max(np.abs(bb_max - bb_min))
    scale_factor = 2.0 / scale
    obj.scale = np.ones(3) * scale_factor
    bb_min *= scale_factor
    bb_max *= scale_factor

    # shift the object up to be resting on z=0
    obj.location[2] -= bb_min[2]
    bb_min[2] = 0
    bb_max[2] -= bb_min[2]
    print_info(obj.location, bb_min, bb_max)

    return obj, (bb_min, bb_max)


def add_cam_tracking_constraint(camera, lookat):
    """Add a tracking constraint so that the camera always points to the object"""
    cam_constraint = camera.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    # Parent the object to the camera
    track_to = bpy.data.objects.new("Empty", None)
    track_to.location = lookat
    camera.parent = track_to

    bpy.context.scene.collection.objects.link(track_to)
    bpy.context.view_layer.objects.active = track_to
    cam_constraint.target = track_to
    return track_to


def add_camera(camera_loc, lookat):
    """Choose a camera view to the scene pointing towards lookat"""
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    print_info("CAMERA LOC", camera_loc)
    camera.location = camera_loc + lookat

    track_to = add_cam_tracking_constraint(camera, lookat)
    bpy.context.view_layer.update()
    return camera, track_to


def add_light_env(filepath, strength=1, rot_vec_rad=(0, 0, 0), scale=(1, 1, 1)):
    """Add an HDRI as the environment map for lighting.
    Can only use if using CYCLES as rendering engine."""
    engine = bpy.context.scene.render.engine
    assert engine == "CYCLES", "Rendering engine is not Cycles"

    bpy.data.images.load(filepath, check_existing=True)
    env = bpy.data.images[osp.basename(filepath)]

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new("ShaderNodeBackground")
    links.new(bg_node.outputs["Background"], nodes["World Output"].inputs["Surface"])

    # Environment map
    texcoord_node = nodes.new("ShaderNodeTexCoord")
    env_node = nodes.new("ShaderNodeTexEnvironment")
    env_node.image = env
    mapping_node = nodes.new("ShaderNodeMapping")
    links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
    links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    bg_node.inputs["Strength"].default_value = strength
    print_info("LIGHT STRENGTH:", strength)


def select_devices(device_type, gpus):
    """If using GPU rendering, select which GPUs to use."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"
    for dev_type in preferences.get_device_types(bpy.context):
        preferences.get_devices_for_type(dev_type[0])
        for device in preferences.devices:
            device.use = False
    preferences.get_devices_for_type(device_type)
    sel_devices = [
        device for device in preferences.devices if device.type == device_type
    ]
    print_info(len(sel_devices), gpus)
    for idx in gpus:
        sel_devices[idx].use = True
    for device in sel_devices:
        device.use = True
        print(
            "Device {} of type {} found, use {}".format(
                device.name, device.type, device.use
            )
        )


def set_cycles(args):
    """Set up PBR rendering with the CYCLES rendering engine.
    More photorealistic, much slower."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cycles = scene.cycles

    cycles.use_progressive_refine = True
    cycles.samples = args.n_samples
    cycles.max_bounces = 8
    cycles.caustics_reflective = True
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 8
    cycles.glossy_bounces = 4
    cycles.volume_bounces = 0

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds["World"]
    world.cycles.sample_as_light = True
    cycles.blur_glossy = 2.0
    cycles.sample_clamp_indirect = 10.0

    world.use_nodes = True

    if args.use_gpu:
        if args.gpus is not None:
            select_devices("CUDA", args.gpus)
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        # XXX the following needs to be called to register preference update
        devices = bpy.context.preferences.addons["cycles"].preferences.get_devices()

    bpy.context.scene.render.use_persistent_data = True
    # so we don't have to recompute the MIS map for the same world layer
    bpy.context.scene.world.cycles.sampling_method = "MANUAL"
    bpy.context.scene.world.cycles.sample_map_resolution = 1024
    bpy.context.scene.view_layers["View Layer"].cycles.use_denoising = True

    scene.render.tile_x = 256 if args.use_gpu else 16
    scene.render.tile_y = 256 if args.use_gpu else 16
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = str(args.color_depth)


def set_eevee(args):
    """Set up the render for the Blender Eevee rendering engine.
    Rendering this way is NOT PBR, but is much faster."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    print_info(scene.render.engine)
    args.render_bg = False

    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = str(args.color_depth)


def hide_objects(obj_names):
    """Hide objects with names given in `object_names`"""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        obj.select_set(obj.name in obj_names)
    for sel in bpy.context.selected_objects:
        sel.hide = True


def delete_objects(obj_names):
    """Delete objects with names given in `object_names`"""
    for obj in bpy.data.objects:
        obj.select_set(obj.name in obj_names)
    bpy.ops.object.delete()

    # Remove meshes, textures, materials, etc to avoid memory leaks.
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    bpy.context.view_layer.update()


def global_setup(args):
    """Set up the scene and lighting to render for all instances in this process."""
    delete_objects([obj.name for obj in bpy.data.objects])

    bpy.context.scene.use_nodes = True

    if args.use_pbr:
        set_cycles(args)
        if args.light_env is not None:
            add_light_env(args.light_env, args.light_strength)
        _add_background_layer()
    else:
        set_eevee(args)
        add_lamps()

    setup_global_render(args)


def setup_scene(args, model_dirs):
    """set up the scene according to the args provided"""
    objs, bbs, locs = [], [], []
    lookat = Vector((0, 0, 0))
    axis_forward = "-Z"
    axis_up = "Y"
    if len(model_dirs) == 1:
        obj, bb = import_object(
            model_dirs[0],
            args.model_path,
            axis_forward=axis_forward,
            axis_up=axis_up,
        )
        obj.location[0] = 0
        obj.location[1] = 0
        lookat = obj.location
        camera_loc = np.array((0, 4.0, lookat[2]))
        objs = [obj]
    elif len(model_dirs) == 2:
        for model_dir in model_dirs:
            obj, bb = import_object(
                model_dir,
                args.model_path,
                axis_forward=axis_forward,
                axis_up=axis_up,
            )
            print_info(obj.name, bb)
            objs.append(obj)
            bbs.append(bb)
        # move objects to quadrant 1 and 3
        sign = -1
        for obj, bb in zip(objs, bbs):
            obj.location[0] = sign * bb[0][0]
            obj.location[1] = sign * bb[0][1]
            sign *= -1
            lookat += obj.location
            print_info("NEW LOCATION", obj.location)
        lookat /= len(objs)
        camera_loc = np.array((0, 6.0, lookat[2]))
    else:
        raise NotImplementedError

    # point camera in between objects added
    camera, track_to = add_camera(camera_loc, lookat)
    view_dist = np.linalg.norm(camera_loc)

    return objs, camera, track_to, view_dist, lookat


def setup_global_render(args):
    """Set the rendering settings over all instances"""
    scene = bpy.context.scene
    scene.render.filepath = "/tmp/{}".format(time())  # throw away the composite

    _add_object_output(scene)

    scene.render.film_transparent = True
    if args.render_bg:  # render bg separately
        _add_background_output(scene)

    if args.render_alpha:
        _add_alpha_output(scene)

    if args.render_depth:
        _add_depth_output(scene)


def _render_single(filepath, camera, args):
    scene = bpy.context.scene
    scene.camera = camera

    file_prefixes = [
        _update_node_filepath(filepath, scene, "Object File Output", "obj"),
    ]
    if args.render_bg:
        file_prefixes.append(
            _update_node_filepath(filepath, scene, "Env File Output", "env")
        )
    if args.render_alpha:
        file_prefixes.append(
            _update_node_filepath(filepath, scene, "Alpha File Output", "alpha")
        )

    if args.render_depth:
        file_prefixes.append(
            _update_node_filepath(filepath, scene, "Depth File Output", "depth")
        )

    bpy.ops.render.render(write_still=True)
    return file_prefixes


def _move_files(dirname, file_prefixes):
    print(file_prefixes)
    # for all the file prefixes, just move them from the blender rendered file the desired name
    for prefix in file_prefixes:
        matching = glob.glob("{}_*".format(osp.join(dirname, prefix)))
        print(matching)
        if len(matching) != 1:
            raise NotImplementedError
        ext = osp.splitext(matching[0])[1]
        output = "{}/{}{}".format(dirname, prefix, ext)
        print(matching[0], output)
        shutil.move(matching[0], output)


def _update_node_filepath(filepath, scene, node_name, prefix):
    outnode = scene.node_tree.nodes[node_name]
    fname = "{}_{}".format(osp.basename(filepath), prefix)
    outnode.base_path = osp.dirname(filepath)
    outnode.file_slots[0].path = fname + "_"
    return fname


def _add_compositing(scene):
    tree = scene.node_tree
    alpha_node = tree.nodes.new("CompositorNodeAlphaOver")
    composite_node = tree.nodes["Composite"]
    tree.links.new(
        tree.nodes["Render Layers"].outputs["Image"], alpha_node.inputs[1]
    )  # image 1
    tree.links.new(
        tree.nodes["Background Render Layers"].outputs["Image"], alpha_node.inputs[2]
    )  # image 2
    tree.links.new(alpha_node.outputs["Image"], composite_node.inputs["Image"])


def _add_object_output(scene):
    result_socket = scene.node_tree.nodes["Render Layers"].outputs["Image"]
    outnode = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    outnode.name = "Object File Output"
    scene.node_tree.links.new(result_socket, outnode.inputs["Image"])


def _add_background_output(scene):
    result_socket = scene.node_tree.nodes["Background Render Layers"].outputs["Env"]
    bg_file_output = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    bg_file_output.name = "Env File Output"
    scene.node_tree.links.new(result_socket, bg_file_output.inputs["Image"])


def _add_alpha_output(scene):
    result_socket = scene.node_tree.nodes["Render Layers"].outputs["Alpha"]
    alpha_file_output = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    alpha_file_output.name = "Alpha File Output"
    scene.node_tree.links.new(result_socket, alpha_file_output.inputs["Image"])


def _add_depth_output(scene):
    result_socket = scene.node_tree.nodes["Render Layers"].outputs["Depth"]
    depth_file_output = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    depth_file_output.name = "Depth File Output"
    depth_file_output.format.file_format = "OPEN_EXR"
    depth_file_output.format.color_depth = "32"
    scene.node_tree.links.new(result_socket, depth_file_output.inputs["Image"])


def _add_background_layer():
    scene = bpy.context.scene
    bpy.ops.scene.view_layer_add()
    print(scene.view_layers.keys())
    new_layer_name = [key for key in scene.view_layers.keys() if key.endswith(".001")][
        0
    ]
    bg_view_layer = scene.view_layers[new_layer_name]
    bg_view_layer.name = "Background Layer"
    bg_view_layer.use_ao = False
    bg_view_layer.use_solid = False
    bg_view_layer.use_strand = False
    bg_view_layer.use_pass_combined = False
    bg_view_layer.use_pass_z = False
    bg_view_layer.use_pass_environment = True

    bpy.context.window.view_layer = scene.view_layers["View Layer"]

    # make new render layers and output node
    bg_render_layers = scene.node_tree.nodes.new(type="CompositorNodeRLayers")
    bg_render_layers.name = "Background Render Layers"
    bg_render_layers.layer = bg_view_layer.name


def render_views(
    args,
    model_dirs,
    rng,
):
    """Render the model with the specified viewpoint."""
    start = time()
    assert len(model_dirs) >= 1
    out_dir = osp.join(args.out_dir, osp.basename(model_dirs[0]))
    print_info(out_dir, osp.isdir(out_dir))
    if (
        osp.isdir(out_dir)
        and len(os.listdir(out_dir)) >= args.n_views
        and not args.overwrite
    ):
        print_info("images already written for {}".format(out_dir))
        return False
    os.makedirs(out_dir, exist_ok=True)
    print_info("saving outputs to {}".format(out_dir))

    start = time()
    objs, camera, track_to, view_dist, lookat = setup_scene(args, model_dirs)
    print_info("VIEW_DIST", view_dist)

    frames = []
    files = []
    pitch_range = [0, np.deg2rad(80)]
    euler_zs = 6 * np.pi * np.arange(args.n_views) / args.n_views
    if args.split == "train":
        # if training, we use binned uniform views around the hemisphere
        # and add bounded random noise to camera location
        euler_xs = rng.uniform(*pitch_range, size=(args.n_views,))
        euler_zs += rng.uniform(np.pi / args.n_views, size=(args.n_views,))
    else:
        # if val or test, we use the Archimedes spiral introduced by SRN
        euler_xs = np.arange(args.n_views) / args.n_views * np.diff(pitch_range)
    for i in range(args.n_views):
        rot_euler = np.array([euler_xs[i], 0, euler_zs[i]])
        track_to.rotation_euler = rot_euler
        filepath = osp.join(out_dir, "view_{:03d}".format(i))
        files.extend(_render_single(filepath, camera, args))

        # NOTE: camera matrix must be written AFTER render because view layer is updated lazily
        camera_matrix = np.array(camera.matrix_world).tolist()
        frame_data = DotMap(transform_matrix=camera_matrix)
        frame_data.file_path = filepath
        frames.append(frame_data)

    _move_files(out_dir, files)
    delete_objects([obj.name for obj in objs])

    out_data = DotMap(frames=frames)
    out_data.model_ids = [osp.basename(name) for name in model_dirs]
    out_data.camera_angle_x = camera.data.angle_x

    with open(osp.join(out_dir, "transforms.json"), "w") as f:
        json.dump(out_data, f, indent=1, separators=(",", ":"))
    delta = time() - start
    print_info("rendering {} took {} seconds".format(model_dirs[0], delta))

    print_info("time to render {}: {}".format(model_dirs[0], time() - start))
    return True


def _load_split_txt(path):
    with open(path, "r") as f:
        return list(map(lambda s: str(s.split()[0]), f.readlines()))


def get_split(args):
    object_dir = args.src_model_dir
    val_frac = args.val_frac
    test_frac = args.test_frac

    models_all = [
        subd for subd in glob.glob("{}/*".format(object_dir)) if osp.isdir(subd)
    ]
    n_total = len(models_all)
    print("total models in {}: {}".format(object_dir, n_total))
    n_val = int(val_frac * n_total)
    n_test = int(test_frac * n_total)
    n_train = n_total - (n_val + n_test)

    train_split_path = osp.join(object_dir, "train_split_{}.txt".format(n_train))
    val_split_path = osp.join(object_dir, "val_split_{}.txt".format(n_val))
    test_split_path = osp.join(object_dir, "test_split_{}.txt".format(n_test))
    if (
        osp.isfile(train_split_path)
        and osp.isfile(val_split_path)
        and osp.isfile(test_split_path)
    ):
        print(
            "splits {}, {}, {} already exist".format(
                train_split_path, val_split_path, test_split_path
            )
        )
    else:
        val_end = n_train + n_val
        permute = np.random.permutation(n_total)
        train_models = [models_all[i] for i in permute[:n_train]]
        val_models = [models_all[i] for i in permute[n_train:val_end]]
        test_models = [models_all[i] for i in permute[val_end:]]

        with open(train_split_path, "w") as f:
            f.write("\n".join(train_models))

        with open(val_split_path, "w") as f:
            f.write("\n".join(val_models))

        with open(test_split_path, "w") as f:
            f.write("\n".join(test_models))

        print(
            "wrote splits to {}, {}, {}".format(
                train_split_path, val_split_path, test_split_path
            )
        )

    if args.split == "train":
        return _load_split_txt(train_split_path)
    elif args.split == "val":
        return _load_split_txt(val_split_path)
    elif args.split == "test":
        return _load_split_txt(test_split_path)
    else:
        raise NotImplementedError


def parse_args():
    # Blender assumes all arguments before ' -- ' are Blender arguments.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    # Object IDs taken from the ShapeNet category JSON
    OBJ_IDS = dict(
        table="04379243",
        chair="03001627",
        mug="03797390",
        bench="02828884",
        lamp="03636649",
        bowl="02880940",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", required=True, help="Where to write the rendered images"
    )
    parser.add_argument(
        "--src_model_dir",
        required=True,
        help="Directory where ShapeNet models are stored",
    )
    parser.add_argument(
        "--object",
        choices=OBJ_IDS.keys(),
        default="chair",
        help="Which ShapeNet class to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_normalized.obj",
        help="Path to model, inside an instance of the ShapeNet class directory",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of instances to use as validation",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of instances to use as test",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to render",
    )
    parser.add_argument(
        "--n_views", type=int, default=20, help="Number of views to render per instance"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If rendering a subset of the instances, starting instance to render.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="If rendering a subset of the instances, ending instance to render.",
    )
    parser.add_argument(
        "--n_objects", type=int, default=1, help="number of objects in scene"
    )

    parser.add_argument("--use_pbr", action="store_true", help="Whether to render with physically based rendering (Blender Cycles) or not.")
    parser.add_argument(
        "--light_env",
        default=None,
        help="If using PBR rendering and an HDRI light map, the path to the HDRI",
    )
    parser.add_argument(
        "--light_strength",
        type=float,
        default=3,
        help="If using HDRI light map, HDRI strength",
    )
    parser.add_argument(
        "--render_alpha", action="store_true", help="select to render the object masks"
    )
    parser.add_argument(
        "--render_depth", action="store_true", help="select to render the depth map"
    )
    parser.add_argument(
        "--render_bg",
        action="store_true",
        help="select to render the background layer",
    )
    parser.add_argument(
        "--res", type=int, default=128, help="Output resolution of images (res x res), default 128"
    )
    parser.add_argument(
        "--n_samples", type=int, default=128, help="Number of anti-aliasing samples, default 128"
    )
    parser.add_argument(
        "--color_depth", type=int, default=16, help="Color depth of images (default 16)"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="number of views to be rendered",
    )
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        help="number of views to be rendered",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing renders")
    parser.add_argument("--pool", action="store_true", default=False, help="Render in parallel. Improves performance.")
    args = parser.parse_args(argv)

    obj_id = OBJ_IDS[args.object]
    args.src_model_dir = osp.join(args.src_model_dir, obj_id)
    args.out_dir = osp.join(
        args.out_dir, "{}_{}obj".format(obj_id, args.n_objects), args.split
    )
    print(args)
    return args


def _main_sequential(args):
    """Render everything in a single process"""
    model_dirs = get_split(args)
    end_idx = args.end_idx if args.end_idx > 0 else len(model_dirs)
    rng = np.random.default_rng(seed=9)
    for model_dir in model_dirs[args.start_idx : end_idx]:
        sel_dirs = [model_dir]
        for _ in range(args.n_objects - 1):
            sel_dirs.append(rng.choice(model_dirs))
        render_views(args, sel_dirs, rng)


def _main_parallel(args):
    """Spawn child processes after global setup to speed up rendering"""
    model_dirs = get_split(args)
    end_idx = args.end_idx if args.end_idx > 0 else len(model_dirs)
    n_instances = end_idx - args.start_idx

    # need to pass in separate RNGs into the child processes.
    seed_gen = SeedSequence(9)
    rngs = [Generator(MT19937(sg)) for sg in seed_gen.spawn(n_instances)]

    futures = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        for model_dir, rng in zip(model_dirs[args.start_idx : end_idx], rngs):
            sel_dirs = [model_dir]
            for _ in range(args.n_objects - 1):
                sel_dirs.append(rng.choice(model_dirs))
            futures.append(
                executor.submit(
                    render_views,
                    args,
                    sel_dirs,
                    rng,
                )
            )
        for future in futures:
            _ = future.result()


def main():
    """Launch rendering.

    Example Usage:
        blender --background --python render_shapenet.py -- --object chair

    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    global_setup(args)

    if args.pool:
        _main_parallel(args)
    else:
        _main_sequential(args)
    print("finished rendering")


if __name__ == "__main__":
    main()

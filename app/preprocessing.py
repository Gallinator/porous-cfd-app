import json
import math
import shutil
from pathlib import Path
import bpy
import numpy as np
from sympy import nsolve, solve
from sympy.abc import x

from app.api_models import Predict2dInput


def particle_size_from_re(re: float, u: float, nu: float) -> list:
    return nsolve(((u * x) / nu - re), x, 0)


def eps_from_d(dp: float, k: float) -> list:
    return solve((dp ** 2 * x ** 3) / (180 * (1 - x) ** 2) - k, x)


def f_from_eps(eps: float) -> float:
    return 1.8 / math.sqrt(180 * eps ** 5) * eps


def get_f_from_d(d: float, u: float, re: float) -> float:
    nu = 1489.4e-6
    k = 1 / d
    dp = particle_size_from_re(re, u, nu)
    eps = eps_from_d(dp, k)[0].n(chop=True)
    return float(f_from_eps(eps) / math.sqrt(k))


def copy_config_with_boundary_conditions(config_path: Path, data_path: Path, input_data: Predict2dInput):
    with open(config_path, 'r') as file:
        config = json.load(file)
    config["cfd params"]["coeffs"][0]["d"] = [input_data.d, input_data.d, 0]
    config["cfd params"]["inlet"] = [input_data.inlet_u]
    config["cfd params"]["angle"] = [input_data.inlet_angle, input_data.inlet_angle, 0]

    re = 0.8 if "pipn" in input_data.model else 13.428
    f = get_f_from_d(input_data.d, 0.2, re)
    config["cfd params"]["coeffs"][0]["f"] = [f, f, 0]

    with open(data_path / "config.json", 'w') as f:
        f.write(json.dumps(config))


def create_session_folders(assets_dir: str, session_dir: str, predict_input: Predict2dInput):
    assets_dir = Path(assets_dir)
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(assets_dir / "openfoam-case-template", session_dir / "assets" / "openfoam-case-template")
    shutil.copy(assets_dir / "data_config.json", session_dir / "assets")
    split_dir = session_dir / "assets" / "meshes" / "split"
    split_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(assets_dir / "transforms.json", split_dir)

    data_dir = session_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)


def path_to_obj(x: list[float], y: list[float], dest_path: str):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh = bpy.data.meshes.new("mesh")
    obj = bpy.data.objects.new("object", mesh)

    coords = [(x, y, 0.05) for (x, y) in zip(x, y)]
    edges = [(i, (i + 1) % len(x)) for i in range(len(x))]
    face = np.arange(len(x))

    mesh.from_pydata(coords, edges, [face])
    mesh.update()

    bpy.context.collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.editmode_toggle()

    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (-0, -0, -0.1),
                                "constraint_axis": (False, False, True)})

    bpy.ops.object.editmode_toggle()

    bpy.ops.wm.obj_export(filepath=f'{dest_path}/mesh.obj',
                          forward_axis='Y',
                          up_axis='Z',
                          export_materials=False,
                          export_selected_objects=True)

    bpy.ops.object.delete()

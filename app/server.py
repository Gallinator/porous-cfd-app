import asyncio
import os.path
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing import get_context
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from scipy.interpolate import griddata
from starlette.staticfiles import StaticFiles
from torch.utils.data import DataLoader

from app.config import AppSettings
from porous_cfd.dataset.foam_data import FoamData
from porous_cfd.dataset.foam_dataset import FoamDataset, collate_fn
from porous_cfd.models.pi_gano.pi_gano import PiGano
from porous_cfd.models.pi_gano.pi_gano_pp import PiGanoPp
from porous_cfd.models.pipn.pipn_foam import PipnFoam, PipnFoamPp, PipnFoamPpMrg
from app.api_models import Predict2dInput, Response2d


def get_interpolation_grid(points, grid_res) -> list[np.ndarray]:
    points_x, points_y = points[:, 0].flatten(), points[:, 1].flatten()
    xx = np.linspace(points_x.min(), points_x.max(), grid_res)
    yy = np.linspace(points_y.min(), points_y.max(), grid_res)
    return np.meshgrid(xx, yy)


def interpolate_on_grid(grid, points, *data) -> list:
    return [griddata(points, d, tuple(grid), method='cubic', fill_value=0).flatten() for d in data]


def ndarrays_to_list(data: dict[str:np.ndarray]):
    for k, v in data.items():
        data[k] = v.flatten().tolist()
    return data


def inverse_transform_output(dataset: FoamDataset, data: FoamData, *fields) -> list[np.ndarray]:
    return [dataset.normalizers[f].inverse_transform(data[f].numpy(force=True)) for f in fields]


def generate_data(input_data: Predict2dInput, session_root: str):
    # Only import blender in a subprocess, as the path is passed from the main process (see https://projects.blender.org/blender/blender/issues/98534)
    # This has to be done here otherwise the context is incorrect
    from porous_cfd.examples.duct_variable_boundary.generator_2d_variable import Generator2DVariable
    from app.preprocessing import path_to_obj, create_session_folders

    create_session_folders("assets", session_root, input_data)

    path_to_obj(input_data.points["x"], input_data.points["y"], f"{session_root}/assets/meshes/split")

    datagen = Generator2DVariable(f"{session_root}/assets", openfoam_cmd, settings.n_procs, 0, False)
    datagen.write_momentum = False
    datagen.save_plots = False

    datagen.generate(f"{session_root}/data")

    # Override meta and min_points from training set
    model_dir = "pipn" if "pipn" in input_data.model else "pi-gano"
    shutil.copy(f"assets/{model_dir}/min_points.json", f"{session_root}/data")
    shutil.copy(f"assets/{model_dir}/meta.json", f"{session_root}/data/split")


def postprocess(dataset: FoamDataset, predicted: FoamData, residuals: FoamData):
    c, tgt_u, tgt_p = inverse_transform_output(dataset, dataset[0], "C", "U", "p")
    points = {"x": c[..., 0],
              "y": c[..., 1]}

    target = {"Ux": tgt_u[..., 0],
              "Uy": tgt_u[..., 1],
              "U": np.linalg.norm(tgt_u, axis=1),
              "p": tgt_p}

    pred_u, pred_p = inverse_transform_output(dataset, predicted, "U", "p")

    pred = {"Ux": pred_u[0, ..., 0],
            "Uy": pred_u[0, ..., 1],
            "U": np.linalg.norm(pred_u[0], axis=1),
            "p": pred_p[0]}

    error_u, error_p = np.abs(pred_u - tgt_u), np.abs(pred_p - tgt_p)
    error = {"Ux": error_u[0, ..., 0],
             "Uy": error_u[0, ..., 1],
             "U": np.linalg.norm(error_u[0], axis=1),
             "p": error_p[0]}

    residuals = {"Momentumx": residuals["Momentumx"].numpy(force=True)[0],
                 "Momentumy": residuals["Momentumy"].numpy(force=True)[0],
                 "Momentum": np.linalg.norm(residuals["Momentum"].numpy(force=True)[0], axis=1),
                 "div": residuals["div"].numpy(force=True)[0]}

    porous_ids = dataset[0]["cellToRegion"].flatten().numpy(force=True)
    grid = get_interpolation_grid(c, 50)
    grid_points = {"x": grid[0].flatten(), "y": grid[1].flatten()}

    grid_pred = interpolate_on_grid(grid, c, pred["Ux"], pred["Uy"], pred["U"], pred["p"])
    grid_pred = dict(zip(pred.keys(), grid_pred))

    grid_target = interpolate_on_grid(grid, c, target["Ux"], target["Uy"], target["U"], target["p"])
    grid_target = dict(zip(target.keys(), grid_target))

    grid_error = interpolate_on_grid(grid, c, error["Ux"], error["Uy"], error["U"], error["p"])
    grid_error = dict(zip(pred.keys(), grid_error))

    internal_c = dataset.normalizers["C"].inverse_transform(dataset[0]["internal"]["C"].numpy(force=True))

    grid_residuals = interpolate_on_grid(grid, internal_c, residuals["Momentumx"],
                                         residuals["Momentumy"],
                                         residuals["Momentum"],
                                         residuals["div"])
    grid_residuals = dict(zip(residuals.keys(), grid_residuals))

    raw_data = Response2d(points=ndarrays_to_list(points),
                          target=ndarrays_to_list(target),
                          porous_ids=porous_ids.tolist(),
                          predicted=ndarrays_to_list(pred),
                          error=ndarrays_to_list(error),
                          residuals=ndarrays_to_list(residuals))

    grid_data = Response2d(points=ndarrays_to_list(grid_points),
                           target=ndarrays_to_list(grid_target),
                           predicted=ndarrays_to_list(grid_pred),
                           error=ndarrays_to_list(grid_error),
                           residuals=ndarrays_to_list(grid_residuals))

    return {"raw_data": raw_data, "grid_data": grid_data}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.models['pipn'] = PipnFoam.load_from_checkpoint("assets/pipn/pipn.ckpt")
    app.models['pipn'].verbose_predict = True
    app.models['pipn_pp'] = PipnFoamPp.load_from_checkpoint("assets/pipn/pipn-pp.ckpt")
    app.models['pipn_pp'].verbose_predict = True
    app.models['pipn_pp_mrg'] = PipnFoamPpMrg.load_from_checkpoint("assets/pipn/pipn-pp-mrg.ckpt")
    app.models['pipn_pp_mrg'].verbose_predict = True
    app.models['pi_gano'] = PiGano.load_from_checkpoint("assets/pi-gano/pi_gano.ckpt")
    app.models['pi_gano'].verbose_predict = True
    app.models['pi_gano_pp'] = PiGanoPp.load_from_checkpoint("assets/pi-gano/pi_gano_pp.ckpt")
    app.models['pi_gano_pp'].verbose_predict = True
    yield
    app.process_pool.shutdown()


settings = AppSettings()
app = FastAPI(lifespan=lifespan)
openfoam_cmd = f'{settings.openfoam_dir}/etc/openfoam'
app.process_pool = ProcessPoolExecutor(mp_context=get_context("forkserver"))
app.models = {}


@app.post("/predict", summary="Predict flow from porous object", response_model=dict[str, Response2d])
async def predict(input_data: Predict2dInput):
    session_dir = f"sessions/{input_data.uuid}"
    try:
        event_loop = asyncio.get_running_loop()
        partial_f = partial(generate_data, input_data, session_dir)
        await event_loop.run_in_executor(app.process_pool, partial_f)

        dataset = FoamDataset(f"{session_dir}/data/split", 1000, 200, 500,
                              np.random.default_rng(8421))

        torch.manual_seed(8421)
        data_loader = DataLoader(dataset,
                                 1,
                                 num_workers=settings.n_procs,
                                 persistent_workers=True,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn)

        trainer = Trainer(logger=False,
                          enable_checkpointing=False,
                          inference_mode=False,
                          callbacks=[RichProgressBar()])

        # No lock as async coroutines run on the main thread
        model = app.models[input_data.model]
        predicted, residuals = trainer.predict(model, dataloaders=data_loader)[0]

        shutil.rmtree(session_dir)

        # Detach predicted data as autograd computation graphs are not supported with multiprocessing
        partial_f = partial(postprocess, dataset, predicted.detach(), residuals.detach())
        return await event_loop.run_in_executor(app.process_pool, partial_f)
    except:
        traceback.print_exc()

        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)

        raise HTTPException(status_code=500)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

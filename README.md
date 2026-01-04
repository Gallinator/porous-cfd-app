---
title: Porous Cfd
emoji: 🌖
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: gpl-3.0
short_description: 'Porous media CFD with PINN '
app_port: 8000
app_file: app/server.py
---

# Porous CFD web app
A small web app to experiment with the 2D models proposed in the paper *Modeling diffusion in complex media through Physically Informed Neural Networks*.<br>
Features:
- Physics Informed PointNet and Physics Informed Geometry Aware Neural Networks available
- PointNet++ variants included
- Simulation of porous objects inside ducts using the Navier-Stokes-Darcy equations
- Variable inlet velocity, angle and porosity
- Porous object editor using B-Splines
- Interactive plots
- Ground truth OpenFOAM simulations
- Parallel server requests processing with FastAPI

The app is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Gallinator/porous-cfd).


![Editor](docs/Editor_screen.png)

![Inference](docs/Inference_screen.png)
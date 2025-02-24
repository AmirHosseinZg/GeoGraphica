from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import time

import matplotlib

matplotlib.use("Agg")

from GeoGraphicaPr.core.computations.compute import compute_gradients
from GeoGraphicaPr.core.plotting import plotter
from GeoGraphicaPr.tests import test_plooter
from GeoGraphicaPr.utils import files_paths
from fastapi.responses import HTMLResponse
from tkinter import filedialog
from typing import Optional
from GeoGraphicaPr.utils import utility
import matplotlib.pyplot as plt
import tkinter as tk
import os

app = FastAPI()

# app.mount("/static", StaticFiles(directory="GeoGraphicaPr/core/my_app/static"), name="static")
static_dir = utility.resource_path("core/my_app/static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

total_map_path = None
egm96_maps_path = None
parker_maps_path = None


class InputData(BaseModel):
    longitude_lower: int
    longitude_upper: int
    latitude_lower: int
    latitude_upper: int
    contour_levels: Optional[int] = None
    colorbar_max: Optional[int] = None
    colorbar_min: Optional[int] = None
    longitude_res: float
    latitude_res: float
    colormap: Optional[str] = None
    save_path: str


@app.get("/")
async def home():
    # return FileResponse("GeoGraphicaPr/core/my_app/templates/index.html")
    index_path = utility.resource_path("core/my_app/templates/index.html")
    return FileResponse(index_path)


@app.get("/colormaps")
async def get_colormaps():
    return {"colormaps": plt.colormaps()}


@app.get("/select_directory")
async def select_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    return {"path": directory}


@app.post("/plot")
async def plot_graph(data: InputData):
    try:
        global parker_maps_path, egm96_maps_path, total_map_path

        # Start time to measure execution time
        start_time = time.time()

        # Get the lower and upper bounds for longitude (converted to integers from user input)
        longitude_lowerbound_int = data.longitude_lower
        longitude_upperbound_int = data.longitude_upper
        latitude_lowerbound_int = data.latitude_lower
        latitude_upperbound_int = data.latitude_upper
        longitude_resolution = data.longitude_res
        latitude_resolution = data.latitude_res
        files_paths.base_path = data.save_path

        # (TxxParker, TxyParker, TxzParker,
        #  TyyParker, TyzParker, TzzParker,
        #  TxxEGM96, TxyEGM96, TxzEGM96,
        #  TyyEGM96, TyzEGM96, TzzEGM96,
        #  TxxTOTAL, TxyTOTAL, TxzTOTAL,
        #  TyyTOTAL, TyzTOTAL, TzzTOTAL,
        #  phi_range_deg, landa_range_deg
        #  ) = compute_gradients(
        #     longitude_lowerbound_int, longitude_upperbound_int,
        #     latitude_lowerbound_int, latitude_upperbound_int,
        #     longitude_resolution, latitude_resolution
        # )

        (TxxParker, TxyParker, TxzParker,
         TyyParker, TyzParker, TzzParker,
         TxxEGM96, TxyEGM96, TxzEGM96,
         TyyEGM96, TyzEGM96, TzzEGM96,
         TxxTOTAL, TxyTOTAL, TxzTOTAL,
         TyyTOTAL, TyzTOTAL, TzzTOTAL,
         phi_range_deg, landa_range_deg
         ) = test_plooter.test()

        TOTAL_matrices = [TxxTOTAL, TxyTOTAL, TxzTOTAL, TyyTOTAL, TyzTOTAL, TzzTOTAL]
        TOTAL_titles = ["Txx Total", "Txy Total", "Txz Total", "Tyy Total", "Tyz Total", "Tzz Total"]
        EGM96_matrices = [TxxEGM96, TxyEGM96, TxzEGM96, TyyEGM96, TyzEGM96, TzzEGM96]
        EGM96_titles = ["Txx EGM96", "Txy EGM96", "Txz EGM96", "Tyy EGM96", "Tyz EGM96", "Tzz EGM96"]
        Parker_matrices = [TxxParker, TxyParker, TxzParker, TyyParker, TyzParker, TzzParker]
        Parker_titles = ["Txx Parker", "Txy Parker", "Txz Parker", "Tyy Parker", "Tyz Parker", "Tzz Parker"]

        # Call plotter function (Final Processing)
        total_map_path = plotter.plot_matrices(
            header_title="Gravity Gradient Total Maps",
            matrices=TOTAL_matrices,
            titles=TOTAL_titles,
            phi_range_deg=phi_range_deg,
            landa_range_deg=landa_range_deg,
            selected_colormap=data.colormap,
            start_time=start_time,
            saved_path=data.save_path,
            contour_levels=data.contour_levels,
            colorbar_min=data.colorbar_min,
            colorbar_max=data.colorbar_max
        )
        egm96_maps_path = plotter.plot_matrices(
            header_title="Gravity Gradient EGM96 Maps",
            matrices=EGM96_matrices,
            titles=EGM96_titles,
            phi_range_deg=phi_range_deg,
            landa_range_deg=landa_range_deg,
            selected_colormap=data.colormap,
            start_time=start_time,
            saved_path=data.save_path,
            contour_levels=data.contour_levels,
            colorbar_min=data.colorbar_min,
            colorbar_max=data.colorbar_max
        )
        parker_maps_path = plotter.plot_matrices(
            header_title="Gravity Gradient Parker Maps",
            matrices=Parker_matrices,
            titles=Parker_titles,
            phi_range_deg=phi_range_deg,
            landa_range_deg=landa_range_deg,
            selected_colormap=data.colormap,
            start_time=start_time,
            saved_path=data.save_path,
            contour_levels=data.contour_levels,
            colorbar_min=data.colorbar_min,
            colorbar_max=data.colorbar_max
        )

        execution_time = time.time() - start_time
        return {
            "message": "Plot generated successfully.",
            "execution_time": execution_time,
            "redirect_url": "/plots"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/plots", response_class=HTMLResponse)
def show_plots():
    html_content = """
    <!DOCTYPE html>
    <html lang="fa">
    <head>
        <meta charset="UTF-8">
        <title>Plot Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #1E1E1E;
                color: #FFD700;
                margin: 0;
                padding: 0;
            }
            .tab {
                overflow: hidden;
                background-color: #333;
                display: flex;
            }
            .tab button {
                background-color: inherit;
                flex: 1;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                color: #fff;
                font-weight: bold;
            }
            .tab button:hover {
                background-color: #111;
            }
            .tab button.active {
                background-color: #FF5733;
            }
            .tabcontent {
                display: none;
                padding: 20px;
            }
            .tabcontent img {
                max-width: 90%;
                border: 2px solid #FFD700;
            }
        </style>
        <script>
        function openTab(evt, tabName) {
            var tabcontent = document.getElementsByClassName("tabcontent");
            for (var i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            var tablinks = document.getElementsByClassName("tablinks");
            for (var i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        window.onload = function() {
            document.getElementById("defaultTab").click();
        }
        </script>
    </head>
    <body>
        <div class="tab">
            <button id="defaultTab" class="tablinks" onclick="openTab(event, 'Total')">Total</button>
            <button class="tablinks" onclick="openTab(event, 'EGM96')">EGM96</button>
            <button class="tablinks" onclick="openTab(event, 'Parker')">Parker</button>
        </div>

        <div id="Total" class="tabcontent">
            <h2>Total Maps</h2>
            <img src="/plots/image/total" alt="Total Map">
        </div>

        <div id="EGM96" class="tabcontent">
            <h2>EGM96 Maps</h2>
            <img src="/plots/image/egm96" alt="EGM96 Map">
        </div>

        <div id="Parker" class="tabcontent">
            <h2>Parker Maps</h2>
            <img src="/plots/image/parker" alt="Parker Map">
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/plots/image/{plot_type}")
def get_plot_image(plot_type: str):
    global total_map_path, egm96_maps_path, parker_maps_path

    if plot_type == "total":
        if total_map_path and os.path.isfile(total_map_path):
            return FileResponse(total_map_path, media_type="image/png", filename="total_map.png")
        else:
            return {"error": "Total map image not found."}
    elif plot_type == "egm96":
        if egm96_maps_path and os.path.isfile(egm96_maps_path):
            return FileResponse(egm96_maps_path, media_type="image/png", filename="egm96_map.png")
        else:
            return {"error": "EGM96 map image not found."}
    elif plot_type == "parker":
        if parker_maps_path and os.path.isfile(parker_maps_path):
            return FileResponse(parker_maps_path, media_type="image/png", filename="parker_map.png")
        else:
            return {"error": "Parker map image not found."}
    else:
        return {"error": f"Unknown plot type: {plot_type}."}


def start_server():
    import threading
    thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "127.0.0.1", "port": 5000})
    thread.daemon = True
    thread.start()

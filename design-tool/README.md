# OpenSpiRobs Design Tool

A GUI tool for designing logarithmic-spiral soft robots (2‑cable / 3‑cable), exporting CAD (STEP/STL) and MuJoCo XML models.

## License
- SPDX: PolyForm-Noncommercial-1.0.0
- Commercial use requires a separate license.

## Requirements
- Python 3.10+ (recommended)
- See `requirements.txt`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python DesignTool.py
```

---

# User Guide

## 1) Quick Start (Step by Step)
1) Open a terminal and enter the tool folder (use your local path):

```bash
cd "<path-to-your-download>/design-tool"
```

2) Create/activate a virtual environment (recommended). If you already have one, skip this:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4) Launch the app:

```bash
python DesignTool.py
```

5) Manual validation (2D / 3D):
- Left panel shows the 2D Sketch
- Drag sliders on the right and confirm the sketch updates
- Click **Generate 3D Model** and verify the 3D view renders

6) Export check:
- **Save 2D Sketch** → PNG/PDF created
- **Export STEP/STL** → STEP/STL created
- **Export XML** → `exports/` contains `robot.xml` + `baselink.stl`

7) Quick export verification:
Check the newest folder under `exports/` contains:
- `baselink.stl`
- `robot.xml`
- STEP/STL file(s)

## 2) Screenshots

![Main UI](docs/images/main-ui.png)
![2D Sketch](docs/images/2d-sketch.png)
![3D Model](docs/images/3d-model.png)
![Parameters Panel](docs/images/params-panel.png)

---

# Parameters (Sliders & Toggles)

## 2D Parameters
- **a (mm)**: spiral scale factor
- **b**: spiral growth rate
- **Δθ (deg)**: angular discretization step
- **θ max (π)**: maximum angle range
- **p**: center-spiral blend ratio

## Fabrication Parameters
- **Elastic Layer/Axis (toggle)**: enable elastic layer/axis in geometry
- **Elastic (%)**: elastic layer thickness ratio
- **Pos_TipHole (%)**: hole position ratio at the tip unit
- **Size_TipHole (mm)**: hole diameter at the tip unit
- **Pos_BaseHole (%)**: hole position ratio at the base unit
- **Size_BaseHole (mm)**: hole diameter at the base unit

## Simulation Parameters
- **Stiffness**: exported to MuJoCo joint stiffness (scaled per unit)
- **Damping**: exported to MuJoCo joint damping (scaled per unit)

## Cable Mode
- **2‑cable / 3‑cable toggle**
  - **2‑cable**: model built by extrusion
  - **3‑cable**: model built by revolution

## 3D Parameters (2‑cable only)
- **Extrusion**: thickness for 2‑cable extrusion
- **ConeAngle1**: primary trimming angle
- **ConeAngle2**: secondary trimming angle

---

# Buttons
- **Generate 3D Model**: rebuilds the 3D solid based on current parameters
- **Reset Parameters**: reset all parameters to defaults
- **Reset 3D View**: reset camera
- **Save 2D Sketch**: exports 2D sketch as PNG/PDF
- **Export STEP/STL**: exports CAD for manufacturing
- **Export XML**: exports MuJoCo XML + baselink STL

---

# 3D Printing Workflow (STEP/STL)
1) Click **Export STEP/STL**
2) Open the exported file in your CAD/CAM tool (e.g., Fusion 360, SolidWorks)
3) Check units (mm), wall thickness, and hole placements
4) Slice and print (TPU recommended)

---

# MuJoCo Simulation Workflow (XML)
1) Click **Export XML**
2) The export folder contains `robot.xml` and `baselink.stl`
3) Open `robot.xml` in MuJoCo and run your simulation
4) Adjust **Stiffness/Damping** sliders if needed, then re-export

---

## Files
- `DesignTool.py`: main GUI
- `xml_generator.py`: MuJoCo XML generation
- `spiral_robot/`: spiral math + helpers

## Notes
- Exports are written to the `exports/` folder when you use the GUI buttons.

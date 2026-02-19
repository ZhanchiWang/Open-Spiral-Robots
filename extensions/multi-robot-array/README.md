# Multi-Robot Array GUI

This extension provides a small PySide6 GUI to generate a circular robot array XML
and open it in MuJoCo's interactive viewer.

## Requirements

- Python environment with:
  - `mujoco`
  - `PySide6`

If you use the repository's `design-tool/.venv`, run:

```powershell
.\design-tool\.venv\Scripts\python -m pip install -r design-tool\requirements.txt
```

## Run

```powershell
.\design-tool\.venv\Scripts\python extensions\multi-robot-array\array_gui.py
```

## Usage

1. Select an input MuJoCo XML (`robot.xml`).
2. Adjust array parameters:
   - `offset_y_mm` (default 80)
   - `count` (default 6)
   - `base_rot_deg` (default 0)
   - `tilt_x_deg` (default -30)
3. Click:
   - `Open MuJoCo Viewer (Input)` to inspect the input XML.
   - `Open MuJoCo Viewer (Array)` to generate the array XML and open it.
4. Click `Export XML` to write the final array XML.

## Output

Array XML is written to:

```
extensions/multi-robot-array/output/robot_array.xml
```

The preview XML used by the viewer is:

```
extensions/multi-robot-array/output/_preview_robot_array.xml
```

## CLI Generator

You can also generate arrays directly with:

```powershell
.\design-tool\.venv\Scripts\python extensions\multi-robot-array\generate_array.py --help
```

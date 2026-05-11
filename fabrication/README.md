# Fabrication

This section describes the basic fabrication workflow for printing SpiRobs designs.

## 1) Export CAD
From the design tool, export either **STL** (mesh) or **STEP** (solid).
- STL is usually sufficient for 3D printing.
- STEP is useful if you need to modify geometry in CAD before printing.

## 2) Split oversized parts
If the exported geometry exceeds your printer build volume, use the splitter generator to divide it into printable segments.

Script:
- `fabrication/splitter_generator.py`

Current scope:
- Supports `STEP/STP` via CadQuery solid intersection.
- Supports `STL` via VTK mesh clipping.
- Splits along one axis (`x`, `y`, or `z`) into multiple parts based on a maximum span.
- Supports manual cut planes via comma-separated cut positions.
- Can check each part against a printer build volume and write a JSON report.
- This first version performs geometric splitting only. It does not yet generate keyed joints, dovetails, pins, or snap-fit connectors.

Example:

```powershell
python fabrication/splitter_generator.py design-tool/exports/spi_rob_20260312.step --axis x --max-span-mm 180
```

Example with STL:

```powershell
python fabrication/splitter_generator.py design-tool/exports/spi_rob_20260312.stl --axis z --max-span-mm 150
```

Example with manual cut positions and build-volume checks:

```powershell
python fabrication/splitter_generator.py design-tool/exports/spi_rob_20260312.step --axis x --cut-positions-mm 80,160 --build-volume-mm 256,256,256
```

Output:
- A sibling folder named `<input>_split/`
- Files named `<input>_part01.step`, `<input>_part02.step`, or STL equivalents
- A `split_report.json` file with each segment's bounds, size, and optional build-volume fit result

Recommended workflow:
- First use `STEP` if you need accurate downstream CAD operations.
- Use `STL` when you only need printable meshes.
- Choose the split axis to match the printer build limit and the robot's least critical structural direction.

## 3) Import into slicer
Open the exported file in your slicer (e.g., **BambuLab Studio**).

## 4) Material
- Recommended material: **TPU**

## 5) Slice and print
Set your slicing parameters (layer height, speed, wall thickness) based on TPU best practices, then slice and print.

## 6) Post-processing
Remove supports if used, and verify dimensions before assembly.

## Next step
The intended next iteration is a joinery-aware splitter that can add alignment and reassembly features such as:
- stepped interfaces
- locating pin holes
- external locking sleeves
- keyed joints for repeatable TPU assembly

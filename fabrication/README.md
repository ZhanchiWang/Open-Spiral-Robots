# Fabrication

This section describes the basic fabrication workflow for printing SpiRobs designs.

## 1) Export CAD
From the design tool, export either **STL** (mesh) or **STEP** (solid).
- STL is usually sufficient for 3D printing.
- STEP is useful if you need to modify geometry in CAD before printing.

## 2) Import into slicer
Open the exported file in your slicer (e.g., **BambuLab Studio**).

## 3) Material
- Recommended material: **TPU**

## 4) Slice and print
Set your slicing parameters (layer height, speed, wall thickness) based on TPU best practices, then slice and print.

## 5) Post-processing
Remove supports if used, and verify dimensions before assembly.

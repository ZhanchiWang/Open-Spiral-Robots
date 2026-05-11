import argparse
import json
import math
from pathlib import Path

import cadquery as cq
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPlaneCollection
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkTriangleFilter
from vtkmodules.vtkFiltersGeneral import vtkClipClosedSurface
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
BOUND_INDEX = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}


def _bounds_to_tuple(bounds):
    return (
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        bounds[4],
        bounds[5],
    )


def _dims_from_bounds(bounds):
    return (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )


def _shape_bounds(shape):
    bbox = shape.BoundingBox()
    return (bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax, bbox.zmin, bbox.zmax)


def _parse_cut_positions(raw_value):
    if not raw_value:
        return []
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _normalize_ranges(bounds, axis, max_span=None, cut_positions=None):
    min_idx, max_idx = BOUND_INDEX[axis]
    axis_min = bounds[min_idx]
    axis_max = bounds[max_idx]
    if axis_max - axis_min <= 1e-9:
        return [(axis_min, axis_max)]

    positions = [axis_min, axis_max]
    if cut_positions:
        for pos in cut_positions:
            if axis_min < pos < axis_max:
                positions.append(pos)
    else:
        if max_span is None or max_span <= 0:
            raise ValueError("Either manual cut positions or a positive --max-span-mm is required")
        total = axis_max - axis_min
        count = max(1, math.ceil(total / max_span))
        step = total / count
        for idx in range(1, count):
            positions.append(axis_min + idx * step)

    positions = sorted(set(round(pos, 9) for pos in positions))
    return [(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]


def _make_cutter_box(bounds, axis, low, high):
    mins = [bounds[0], bounds[2], bounds[4]]
    maxs = [bounds[1], bounds[3], bounds[5]]
    axis_id = AXIS_INDEX[axis]
    mins[axis_id] = low
    maxs[axis_id] = high

    lengths = [maxs[i] - mins[i] for i in range(3)]
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    return cq.Workplane("XY").box(
        max(lengths[0], 1e-6),
        max(lengths[1], 1e-6),
        max(lengths[2], 1e-6),
        centered=(True, True, True),
    ).translate(center)


def _make_planes(bounds, axis, low, high):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if axis == "x":
        xmin, xmax = low, high
    elif axis == "y":
        ymin, ymax = low, high
    else:
        zmin, zmax = low, high

    planes = vtkPlaneCollection()
    specs = [
        ((xmin, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((xmax, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, ymin, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, ymax, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, zmin), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, zmax), (0.0, 0.0, -1.0)),
    ]
    for origin, normal in specs:
        plane = vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        planes.AddItem(plane)
    return planes


def _build_report_entry(index, axis, low, high, bounds, output_path, build_volume=None):
    size_x, size_y, size_z = _dims_from_bounds(bounds)
    entry = {
        "part_index": index,
        "axis": axis,
        "range_mm": [low, high],
        "bounds_mm": {
            "xmin": bounds[0],
            "xmax": bounds[1],
            "ymin": bounds[2],
            "ymax": bounds[3],
            "zmin": bounds[4],
            "zmax": bounds[5],
        },
        "size_mm": {
            "x": size_x,
            "y": size_y,
            "z": size_z,
        },
        "output": str(output_path),
    }
    if build_volume is not None:
        fits = size_x <= build_volume[0] and size_y <= build_volume[1] and size_z <= build_volume[2]
        entry["fits_build_volume"] = fits
        entry["build_volume_mm"] = {
            "x": build_volume[0],
            "y": build_volume[1],
            "z": build_volume[2],
        }
    return entry


def split_step(input_path, output_dir, axis, ranges, build_volume=None):
    model = cq.importers.importStep(str(input_path))
    shape = model.val()
    source_bounds = _shape_bounds(shape)

    written = []
    report = []
    for idx, (low, high) in enumerate(ranges, start=1):
        cutter = _make_cutter_box(source_bounds, axis, low, high)
        part = shape.intersect(cutter.val())
        part_bounds = _shape_bounds(part)
        dims = _dims_from_bounds(part_bounds)
        if max(dims) <= 1e-6:
            continue
        out_path = output_dir / f"{input_path.stem}_part{idx:02d}.step"
        cq.exporters.export(part, str(out_path))
        written.append(out_path)
        report.append(_build_report_entry(idx, axis, low, high, part_bounds, out_path, build_volume))
    return written, report, source_bounds


def split_stl(input_path, output_dir, axis, ranges, build_volume=None):
    reader = vtkSTLReader()
    reader.SetFileName(str(input_path))
    reader.Update()

    clean = vtkCleanPolyData()
    clean.SetInputConnection(reader.GetOutputPort())
    clean.Update()
    source_bounds = _bounds_to_tuple(clean.GetOutput().GetBounds())

    written = []
    report = []
    for idx, (low, high) in enumerate(ranges, start=1):
        planes = _make_planes(source_bounds, axis, low, high)
        clip = vtkClipClosedSurface()
        clip.SetClippingPlanes(planes)
        clip.SetInputConnection(clean.GetOutputPort())
        clip.Update()

        tri = vtkTriangleFilter()
        tri.SetInputConnection(clip.GetOutputPort())
        tri.Update()
        poly = tri.GetOutput()
        if poly is None or poly.GetNumberOfCells() == 0:
            continue

        part_bounds = _bounds_to_tuple(poly.GetBounds())
        dims = _dims_from_bounds(part_bounds)
        if max(dims) <= 1e-6:
            continue

        out_path = output_dir / f"{input_path.stem}_part{idx:02d}.stl"
        writer = vtkSTLWriter()
        writer.SetFileName(str(out_path))
        writer.SetInputData(poly)
        writer.Write()
        written.append(out_path)
        report.append(_build_report_entry(idx, axis, low, high, part_bounds, out_path, build_volume))
    return written, report, source_bounds


def _format_build_volume(raw_value):
    if not raw_value:
        return None
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--build-volume-mm must be three comma-separated numbers, for example 256,256,256")
    return tuple(values)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Split exported SpiRobs STEP/STL geometry into printable segments."
    )
    parser.add_argument("input", type=Path, help="Path to input STEP or STL file.")
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis used for splitting. Default: x",
    )
    parser.add_argument(
        "--max-span-mm",
        type=float,
        default=None,
        help="Maximum allowed segment length along the split axis, in mm.",
    )
    parser.add_argument(
        "--cut-positions-mm",
        type=str,
        default="",
        help="Manual cut positions in mm along the chosen axis, comma-separated. Example: 80,160,240",
    )
    parser.add_argument(
        "--build-volume-mm",
        type=str,
        default="",
        help="Printer build volume in mm as x,y,z. Adds fit checks to the report.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path to write a JSON split report. Default: <output-dir>/split_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for split parts. Default: sibling folder named <input>_split",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cut_positions = _parse_cut_positions(args.cut_positions_mm)
    if not cut_positions and (args.max_span_mm is None or args.max_span_mm <= 0):
        raise ValueError("Provide either --max-span-mm or --cut-positions-mm")
    if args.max_span_mm is not None and args.max_span_mm <= 0:
        raise ValueError("--max-span-mm must be positive")

    output_dir = args.output_dir.resolve() if args.output_dir else input_path.with_name(f"{input_path.stem}_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    build_volume = _format_build_volume(args.build_volume_mm)

    suffix = input_path.suffix.lower()
    if suffix in {".step", ".stp"}:
        source_model = cq.importers.importStep(str(input_path))
        source_bounds = _shape_bounds(source_model.val())
        ranges = _normalize_ranges(source_bounds, args.axis, args.max_span_mm, cut_positions)
        written, report, source_bounds = split_step(input_path, output_dir, args.axis, ranges, build_volume)
    elif suffix == ".stl":
        reader = vtkSTLReader()
        reader.SetFileName(str(input_path))
        reader.Update()
        source_bounds = _bounds_to_tuple(reader.GetOutput().GetBounds())
        ranges = _normalize_ranges(source_bounds, args.axis, args.max_span_mm, cut_positions)
        written, report, source_bounds = split_stl(input_path, output_dir, args.axis, ranges, build_volume)
    else:
        raise ValueError("Only STEP/STP and STL inputs are supported")

    if not written:
        raise RuntimeError("No split parts were produced. Check axis and cut settings.")

    report_path = args.report_json.resolve() if args.report_json else output_dir / "split_report.json"
    report_payload = {
        "input": str(input_path),
        "axis": args.axis,
        "source_bounds_mm": {
            "xmin": source_bounds[0],
            "xmax": source_bounds[1],
            "ymin": source_bounds[2],
            "ymax": source_bounds[3],
            "zmin": source_bounds[4],
            "zmax": source_bounds[5],
        },
        "cut_positions_mm": cut_positions,
        "ranges_mm": [[low, high] for (low, high) in ranges],
        "build_volume_mm": None if build_volume is None else {"x": build_volume[0], "y": build_volume[1], "z": build_volume[2]},
        "parts": report,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(written)} part(s) to {output_dir}")
    print(f"Report: {report_path}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

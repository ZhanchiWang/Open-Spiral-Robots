from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Iterable, List, Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class CadExportResult:
    step_path: str
    stl_path: str


def build_solid(polygons: Iterable[List[Point2D]], thickness: float):
    import cadquery as cq

    solid = None
    for poly in polygons:
        if len(poly) < 3:
            continue
        wp = cq.Workplane("XY").polyline(poly).close().extrude(thickness)
        solid = wp if solid is None else solid.union(wp)
    if solid is None:
        raise ValueError("No valid polygons provided for CAD export.")
    return solid


def export_cad(
    polygons: Iterable[List[Point2D]],
    thickness: float,
    out_dir: str,
    prefix: str = "spi_rob",
) -> CadExportResult:
    import cadquery as cq

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_path = os.path.join(out_dir, f"{prefix}_{ts}.step")
    stl_path = os.path.join(out_dir, f"{prefix}_{ts}.stl")

    solid = build_solid(polygons, thickness)
    cq.exporters.export(solid, step_path)
    cq.exporters.export(solid, stl_path)

    return CadExportResult(step_path=step_path, stl_path=stl_path)

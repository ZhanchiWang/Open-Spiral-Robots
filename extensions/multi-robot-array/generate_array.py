"""Generate a circular array of robots from a single MuJoCo XML."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def _latest_robot_xml(repo_root: Path) -> Path | None:
    exports = repo_root / "design-tool" / "exports"
    if not exports.exists():
        return None
    candidates = list(exports.glob("xml_*/*robot.xml")) + list(exports.glob("xml_*/*robot_array.xml"))
    if not candidates:
        candidates = list(exports.glob("xml_*/*robot.xml"))
    if not candidates:
        # fallback: any robot.xml under exports
        candidates = list(exports.rglob("robot.xml"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _prefix_names(elem: ET.Element, prefix: str) -> None:
    # Prefix name attribute if present
    name = elem.get("name")
    if name:
        elem.set("name", f"{prefix}{name}")

    # Prefix references to sites/geoms/tendons if present
    for attr in ("site", "geom", "tendon", "joint", "body"):
        ref = elem.get(attr)
        if ref:
            elem.set(attr, f"{prefix}{ref}")

    # Recurse
    for child in elem:
        _prefix_names(child, prefix)


def _rotation_quat_z(deg: float) -> str:
    rad = math.radians(deg)
    c = math.cos(rad / 2.0)
    s = math.sin(rad / 2.0)
    return f"{c:.9f} 0 0 {s:.9f}"

def _rotation_quat_x(deg: float) -> str:
    rad = math.radians(deg)
    c = math.cos(rad / 2.0)
    s = math.sin(rad / 2.0)
    return f"{c:.9f} {s:.9f} 0 0"

def _quat_mul(q1: str, q2: str) -> str:
    w1, x1, y1, z1 = (float(v) for v in q1.split())
    w2, x2, y2, z2 = (float(v) for v in q2.split())
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return f"{w:.9f} {x:.9f} {y:.9f} {z:.9f}"

    rad = math.radians(deg)
    c = math.cos(rad / 2.0)
    s = math.sin(rad / 2.0)
    return f"{c:.9f} 0 0 {s:.9f}"


def _rotate_xy(x: float, y: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return (x * c - y * s, x * s + y * c)


def build_array(
    input_xml: Path,
    output_xml: Path,
    offset_y_mm: float,
    count: int,
    base_rot_deg: float,
    tilt_x_deg: float,
) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    tendon = root.find("tendon")
    actuator = root.find("actuator")

    if worldbody is None:
        raise RuntimeError("Input XML missing <worldbody>")

    # Create new XML root
    new_root = copy.deepcopy(root)

    # Reset worldbody to keep only non-body elements (ground, light, etc.)
    new_worldbody = new_root.find("worldbody")
    if new_worldbody is None:
        raise RuntimeError("Output XML missing <worldbody>")

    for child in list(new_worldbody):
        if child.tag == "body":
            new_worldbody.remove(child)

    # Identify top-level body(ies) to duplicate
    base_bodies = [child for child in worldbody if child.tag == "body"]

    # Prepare tendon/actuator containers
    new_tendon = new_root.find("tendon")
    new_actuator = new_root.find("actuator")
    if new_tendon is None and tendon is not None:
        new_tendon = ET.SubElement(new_root, "tendon")
    if new_actuator is None and actuator is not None:
        new_actuator = ET.SubElement(new_root, "actuator")
    if new_tendon is not None:
        for child in list(new_tendon):
            new_tendon.remove(child)
    if new_actuator is not None:
        for child in list(new_actuator):
            new_actuator.remove(child)

    # Normalize mesh file paths and copy meshes next to output
    assets = new_root.find("asset")
    if assets is not None:
        for mesh in assets.findall("mesh"):
            file_attr = mesh.get("file")
            if not file_attr:
                continue
            src = (input_xml.parent / file_attr).resolve()
            dst = (output_xml.parent / Path(file_attr).name).resolve()
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    if dst != src:
                        dst.write_bytes(src.read_bytes())
                mesh.set("file", dst.name)
            except Exception:
                # If copy fails, keep original path
                mesh.set("file", file_attr)

    # Base position from first body (if provided)
    base_pos = "0 0 0"
    if base_bodies:
        base_pos = base_bodies[0].get("pos", "0 0 0")
    # Build array
    offset_y_m = offset_y_mm * 0.001
    base_tree = ET.SubElement(new_worldbody, "body")
    base_tree.set("name", "array_base")
    base_tree.set("pos", base_pos)
    # add free joint so array_base is draggable in MuJoCo
    base_tree.append(ET.fromstring('<joint name="array_base_free" type="free"/>'))
    
    base_tree.append(ET.fromstring('<geom name="array_base_marker" type="box" size="0.0025 0.0025 0.0025" rgba="0.2 0.6 0.9 0.5" contype="0" conaffinity="0"/>'))

    for idx in range(count):
        prefix = f"r{idx + 1}_"
        angle = base_rot_deg + idx * (360.0 / count)
        # start at (0, offset_y, 0) in world, then rotate around world Z
        px, py = _rotate_xy(0.0, offset_y_m, angle)

        qz = _rotation_quat_z(angle)
        qx = _rotation_quat_x(tilt_x_deg)

        bx, by, bz = (float(v) for v in base_pos.split())
        wx = px - bx
        wy = py - by
        wz = 0.0 - bz

        wrapper = ET.SubElement(base_tree, "body")
        wrapper.set("name", f"{prefix}base")
        wrapper.set("pos", f"{wx:.9f} {wy:.9f} {wz:.9f}")
        wrapper.set("quat", qz)

        for idx_body, body in enumerate(base_bodies):
            cloned = copy.deepcopy(body)
            if idx_body == 0:
                # apply local X tilt to first link only
                base_quat = cloned.get("quat", "1 0 0 0")
                tilted = _quat_mul(base_quat, _rotation_quat_x(tilt_x_deg))
                cloned.set("quat", tilted)
            _prefix_names(cloned, prefix)
            wrapper.append(cloned)

        # Copy tendons/actuators with prefix
        if tendon is not None and new_tendon is not None:
            for t in tendon:
                t_clone = copy.deepcopy(t)
                _prefix_names(t_clone, prefix)
                new_tendon.append(t_clone)
        if actuator is not None and new_actuator is not None:
            for a in actuator:
                a_clone = copy.deepcopy(a)
                _prefix_names(a_clone, prefix)
                new_actuator.append(a_clone)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(new_root).write(output_xml, encoding="utf-8", xml_declaration=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = _latest_robot_xml(repo_root)

    parser = argparse.ArgumentParser(description="Generate a circular robot array XML")
    parser.add_argument("--input", type=str, default=str(default_input) if default_input else "",
                        help="Input robot.xml (default: latest export)")
    parser.add_argument("--output", type=str, default="extensions/multi-robot-array/output/robot_array.xml",
                        help="Output robot_array.xml")
    parser.add_argument("--offset-y-mm", type=float, default=80, help="World Y translation (mm)")
    parser.add_argument("--count", type=int, default=6, help="Number of robots in array")
    parser.add_argument("--base-rot-deg", type=float, default=0.0,
                        help="Array rotation offset around Z (deg)")
    parser.add_argument("--tilt-x-deg", type=float, default=-30,
                        help="Local X-axis tilt after placement (deg)")

    args = parser.parse_args()
    if not args.input:
        raise SystemExit("No input XML found. Use --input to specify robot.xml")

    build_array(Path(args.input), Path(args.output), args.offset_y_mm, args.count, args.base_rot_deg, args.tilt_x_deg)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

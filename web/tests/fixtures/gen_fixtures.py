"""Generate parity fixtures from the desktop tool's own code.

Pulls the pure geometry functions and the update_2d / _elastic_unit_polys /
_joint_filler_defaults methods out of design-tool/DesignTool.py with `ast`
(so PySide6 / VTK / CadQuery are not needed), runs them on a stub `self`,
and runs xml_generator.generate_mujoco_xml unchanged.

    python3 web/tests/fixtures/gen_fixtures.py > web/tests/fixtures/python.json
"""
import ast
import json
import math
import os
import sys
import tempfile
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.normpath(os.path.join(HERE, "..", "..", "..", "design-tool"))
sys.path.insert(0, TOOL)
from xml_generator import generate_mujoco_xml  # noqa: E402

FUNCS = {
    "_polar_to_cart", "_cart_to_polar", "_reflect_point_across_line",
    "_line_segment_intersection", "_build_unfold_polygons", "_build_polar_units",
}
METHODS = {"update_2d", "_elastic_unit_polys", "_joint_filler_defaults"}

src = open(os.path.join(TOOL, "DesignTool.py"), encoding="utf-8").read()
tree = ast.parse(src)
body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
        body.append(node)
    if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in METHODS:
                body.append(item)
ns = {"math": math, "List": List, "Tuple": Tuple, "Point2D": Tuple[float, float]}
exec(compile(ast.Module(body=body, type_ignores=[]), "DesignTool.py", "exec"), ns)


class Box:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v

    def isChecked(self):
        return bool(self.v)

    def __getattr__(self, _name):
        return lambda *a, **k: None


class Params:
    pass


class Dummy:
    def __getattr__(self, _name):
        return self

    def __call__(self, *a, **k):
        return self


class Stub:
    def __init__(self, cfg):
        self.params = Params()
        self._joint_spins = []
        for key, val in cfg.items():
            setattr(self, key, Box(val))
        self.filler_auto_check = None  # skip the auto-sync branch

    def __getattr__(self, name):
        return Dummy()


for name in METHODS:
    setattr(Stub, name, ns[name])


def r(x):
    return [list(p) for p in x] if x is not None else None


def run(case):
    s = Stub(case["spins"])
    s.update_2d()
    polar = s._last_polar
    ext = case["spins"]["extrusion_spin"]
    s.extrusion_spin = Box(ext)
    # The real update_2d clamps the extrusion box into [0.2, 1.0] * base_size.
    lo = max(0.0, s._base_size * 0.2)
    hi = max(lo, s._base_size)
    s.extrusion_spin = Box(max(lo, min(hi, ext)))
    polys, mirrors = s._elastic_unit_polys()
    widths, heights = s._joint_filler_defaults()
    return {
        "name": case["name"],
        "units_polar": len(polar[3]),
        "theta_len": len(polar[0]),
        "primary": [r(p) for p in s._polys_primary],
        "mirror": [r(p) for p in s._polys_mirror],
        "unitsMirror": [[list(t), list(rr)] for t, rr in polar[4]],
        "elastic": r(s._elastic_poly),
        "ray": [list(s._ray_start), list(s._ray_upper_end), list(s._ray_lower_end)],
        "baseSize": s._base_size,
        "tipSize": s._tip_size,
        "taperDeg": s._taper_angle_deg,
        "length": s._robot_length,
        "elasticUnits": [r(p) for p in polys],
        "fillerWidths": widths,
        "fillerHeights": heights,
    }


def spins(**kw):
    base = dict(
        a_spin=4.95, b_spin=0.1764, dtheta_spin=30, theta_max_spin=6.0, p_spin=0.5,
        elastic_spin=5.0, elastic_check=True, tip_hole_pos_spin=50.0, tip_hole_size_spin=1.4,
        base_hole_pos_spin=90.0, base_hole_size_spin=3.0, cable3_cut_check=True,
        cable3_cut_pos_spin=0.0, cable3_cut_size_spin=14.0, cable_count_spin=3,
        teardrop_check=False, sim_stiffness_spin=5.0, sim_damping_spin=0.2,
        sim_gravity_spin=0.0, bend_dir_spin=0.0, extrusion_spin=5.0,
    )
    base.update(kw)
    return base


cases = [
    {"name": "default", "spins": spins()},
    {"name": "fine", "spins": spins(a_spin=2.3, b_spin=0.12, dtheta_spin=17, theta_max_spin=7.3, p_spin=0.31, elastic_spin=40.0)},
    {"name": "coarse_no_elastic", "spins": spins(a_spin=12.0, b_spin=0.3, dtheta_spin=45, theta_max_spin=4.4, p_spin=0.12, elastic_check=False, extrusion_spin=100.0)},
    {"name": "short", "spins": spins(theta_max_spin=2.4, dtheta_spin=10)},
]

out = {"designs": [run(c) for c in cases], "xml": []}

xml_cases = [
    dict(name="hinge2", unit_height=7.25, scale=1.0967, num_units=6, joint_type="hinge", joint_limit_deg=30,
         robot_length=80.5, site_points=(70.1, 3.2, 79.9, 4.1), cable_mode=2, joint_stiffness=5.0,
         joint_damping=0.2, gravity=0.0, joint_angles=[5, -10, 0, 20.5, 3], bend_dir_deg=0.0,
         tendon_configs=[{"gear": 50.0, "ctrl_lo": -1.0, "ctrl_hi": 1.0, "joint": 5},
                         {"gear": 12.5, "ctrl_lo": -0.5, "ctrl_hi": 2.0, "joint": 3}]),
    dict(name="ball4", unit_height=3.1, scale=1.05, num_units=9, joint_type="ball", joint_limit_deg=20,
         robot_length=40.0, site_points=(36.0, 1.0, 40.0, 1.3), cable_mode=4, joint_stiffness=1.25,
         joint_damping=0.05, gravity=9.81, joint_angles=[0, 0, 4, 4, 4, 0, -2, 1], bend_dir_deg=37.0,
         tendon_configs=None),
]
for c in xml_cases:
    name = c.pop("name")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "robot.xml")
        generate_mujoco_xml(path, stl_name="baselink.stl", **c)
        out["xml"].append({"name": name, "args": c, "xml": open(path, encoding="utf-8").read()})

json.dump(out, sys.stdout, indent=1)

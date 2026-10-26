"""Volumes of the desktop tool's CadQuery solids, for tests/cad.test.ts.

Runs the real _build_cad_parts (and the methods it calls) from
design-tool/DesignTool.py on a stub `self`; needs cadquery==2.6.1:

    python web/tests/fixtures/gen_cad_fixtures.py > web/tests/fixtures/cad_python.json
"""
import ast
import json
import math
import os
import sys
from typing import List, Tuple

import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.normpath(os.path.join(HERE, "..", "..", "..", "design-tool"))

FUNCS = {
    "_polar_to_cart", "_cart_to_polar", "_reflect_point_across_line",
    "_line_segment_intersection", "_build_unfold_polygons", "_build_polar_units",
    "_rodrigues", "_rotate_point_about_axis",
}
METHODS = {
    "update_2d", "_elastic_unit_polys", "_joint_filler_defaults", "_build_cad_parts",
    "_build_frustum_solid", "_build_cable_holes", "_build_cone2_preview_solid",
    "_build_cable3_cut_solid", "_build_joint_fillers", "_apply_joint_pose",
    "_joint_pose_angles", "_halfspace_box", "_cable_angles",
}
tree = ast.parse(open(os.path.join(TOOL, "DesignTool.py"), encoding="utf-8").read())
body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
        body.append(node)
    if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in METHODS:
                item.decorator_list = []
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


class Dummy:
    def __getattr__(self, _name):
        return self

    def __call__(self, *a, **k):
        return self


class P:
    pass


class Stub:
    def __getattr__(self, name):
        return Dummy()


for name in METHODS:
    fn = ns[name]
    setattr(Stub, name, staticmethod(fn) if name == "_halfspace_box" else fn)


def make(cfg):
    s = Stub()
    s.params = P()
    s.params.two_cable = cfg.get("two_cable", True)
    s._joint_spins = []
    s.filler_auto_check = None
    spins = dict(
        a_spin=4.95, b_spin=0.1764, dtheta_spin=30, theta_max_spin=6.0, p_spin=0.5,
        elastic_spin=5.0, elastic_check=True, tip_hole_pos_spin=50.0, tip_hole_size_spin=1.4,
        base_hole_pos_spin=90.0, base_hole_size_spin=3.0, cable3_cut_check=True,
        cable3_cut_pos_spin=0.0, cable3_cut_size_spin=14.0, cable_count_spin=3,
        teardrop_check=False, sim_stiffness_spin=5.0, sim_damping_spin=0.2,
        sim_gravity_spin=0.0, bend_dir_spin=0.0, extrusion_spin=1.0,
        cone1_spin=5.0, cone2_spin=15.0,
    )
    spins.update(cfg.get("spins", {}))
    for k, v in spins.items():
        setattr(s, k, Box(v))
    s.update_2d()
    ext = s._base_size * 0.6  # first-run default extrusion
    s.extrusion_spin = Box(ext)
    c1max = math.degrees(2.0 * math.atan((ext * 0.5) / s._robot_length))
    s.cone1_spin = Box(min(spins["cone1_spin"], c1max))
    s.cone2_spin = Box(min(spins["cone2_spin"], 4.0 * s._taper_angle_deg))
    n_j = len(s._polys_primary) - 1
    angles = cfg.get("angles", lambda i: 0.0)
    s._joint_spins = [Box(angles(i)) for i in range(n_j)]
    w, h = s._joint_filler_defaults()
    s._joint_filler_widths = [Box(v) for v in w]
    s._joint_filler_heights = [Box(v) for v in h]
    return s


def vol(wp):
    if wp is None:
        return 0.0
    return wp.val().Volume()


cases = {
    "2cable": {},
    "3cable": {"two_cable": False, "spins": {"theta_max_spin": 4.0}},
    "4cable_teardrop": {"two_cable": False, "spins": {"theta_max_spin": 4.0, "cable_count_spin": 4, "teardrop_check": True}},
    "2cable_posed": {"spins": {"theta_max_spin": 3.6}, "angles": lambda i: 8.0 if i % 2 else 12.0},
    "3cable_posed": {"two_cable": False, "spins": {"theta_max_spin": 3.6, "bend_dir_spin": 30.0}, "angles": lambda i: 8.0 if i % 2 else 12.0},
}
out = {}
for name, cfg in cases.items():
    s = make(cfg)
    main, elastic = s._build_cad_parts()
    out[name] = {"main": vol(main), "elastic": vol(elastic), "units": len(s._polys_primary)}
    print(name, out[name], file=sys.stderr)
json.dump(out, sys.stdout, indent=1)

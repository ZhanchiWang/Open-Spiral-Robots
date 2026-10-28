import math
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from PySide6.QtGui import QSurfaceFormat, QPainter
from PySide6.QtCore import QMarginsF
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolygon, vtkPolyData, vtkPlane, vtkPlaneCollection
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataNormals, vtkDecimatePro
from vtkmodules.vtkFiltersSources import vtkPlaneSource, vtkArrowSource, vtkCubeSource, vtkSphereSource
from vtkmodules.vtkFiltersModeling import vtkLinearExtrusionFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkClipClosedSurface
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkLegendBoxActor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
try:
    from vtkmodules.qt.QVTKOpenGLNativeWidget import QVTKOpenGLNativeWidget
except Exception:
    QVTKOpenGLNativeWidget = None
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


Point2D = Tuple[float, float]


@dataclass
class Params:
    a: float = 4.95
    b: float = 0.1764
    dtheta_deg: int = 30
    theta_max_pi: float = 6.0
    p: float = 0.5
    elastic_percent: float = 5.0
    elastic_enabled: bool = True
    extrusion: float = 1.0
    cone_angle1: float = 5.0
    cone_angle2: float = 15.0
    tip_hole_pos: float = 50.0
    tip_hole_size: float = 1.4
    base_hole_pos: float = 90.0
    base_hole_size: float = 3.0
    sim_stiffness: float = 5.0
    sim_damping: float = 0.2
    gravity: float = 0.0
    pose_enabled: bool = False
    filler_auto: bool = True
    bend_dir_deg: float = 0.0
    two_cable: bool = True
    num_cables: int = 3
    teardrop_holes: bool = False
    cable3_cut_enabled: bool = True
    cable3_cut_pos: float = 0.0
    cable3_cut_size: float = 14.0


class ToggleSwitch(QWidget):
    toggled = Signal(bool)
    def __init__(
        self,
        checked: bool = True,
        parent: QWidget | None = None,
        on_color: str = "#2f6fb8",
        off_color: str = "#cfd5dd",
    ) -> None:
        super().__init__(parent)
        self.setFixedSize(46, 24)
        self._checked = checked
        self._on_color = on_color
        self._off_color = off_color
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        self._checked = bool(value)
        self.update()

    def mousePressEvent(self, event) -> None:
        self._checked = not self._checked
        self.update()
        self.toggled.emit(self._checked)

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter, QColor

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QColor(self._on_color) if self._checked else QColor(self._off_color)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, self.width(), self.height(), 12, 12)
        knob_x = self.width() - 22 if self._checked else 2
        p.setBrush(QColor("#f8f9fb"))
        p.drawEllipse(knob_x, 2, 20, 20)


def _polar_to_cart(theta: float, r: float) -> Point2D:
    return (r * math.cos(theta), r * math.sin(theta))


def _cart_to_polar(x: float, y: float) -> Point2D:
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2.0 * math.pi
    return (theta, r)


def _reflect_point_across_line(p: Point2D, a: Point2D, b: Point2D) -> Point2D:
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return p
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    projx = ax + t * vx
    projy = ay + t * vy
    return (2.0 * projx - px, 2.0 * projy - py)


def _line_segment_intersection(
    a0: Point2D, a1: Point2D, b0: Point2D, b1: Point2D
) -> Point2D | None:
    ax, ay = a0
    bx, by = a1
    cx, cy = b0
    dx, dy = b1
    r_x = bx - ax
    r_y = by - ay
    s_x = dx - cx
    s_y = dy - cy
    denom = r_x * s_y - r_y * s_x
    if abs(denom) < 1e-12:
        return None
    t = ((cx - ax) * s_y - (cy - ay) * s_x) / denom
    u = ((cx - ax) * r_y - (cy - ay) * r_x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (ax + t * r_x, ay + t * r_y)
    return None


def _rodrigues(
    px: float, py: float, pz: float,
    ux: float, uy: float, uz: float,
    angle_deg: float,
) -> Tuple[float, float, float]:
    """Rotate vector (px,py,pz) about unit axis (ux,uy,uz) by angle_deg."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    dot = ux * px + uy * py + uz * pz
    rx = px * c + (uy * pz - uz * py) * s + ux * dot * (1.0 - c)
    ry = py * c + (uz * px - ux * pz) * s + uy * dot * (1.0 - c)
    rz = pz * c + (ux * py - uy * px) * s + uz * dot * (1.0 - c)
    return (rx, ry, rz)


def _rotate_point_about_axis(
    p: Tuple[float, float, float],
    axis_pt: Tuple[float, float, float],
    axis_dir: Tuple[float, float, float],
    angle_deg: float,
) -> Tuple[float, float, float]:
    rx, ry, rz = _rodrigues(
        p[0] - axis_pt[0], p[1] - axis_pt[1], p[2] - axis_pt[2],
        axis_dir[0], axis_dir[1], axis_dir[2],
        angle_deg,
    )
    return (rx + axis_pt[0], ry + axis_pt[1], rz + axis_pt[2])


def _build_unfold_polygons(
    a: float,
    b: float,
    dtheta_deg: int,
    p: float,
    turns: float,
    unit_count: int,
) -> Tuple[List[List[Point2D]], List[List[Point2D]]]:
    dtheta = math.radians(max(1, int(dtheta_deg)))
    gamma = math.exp(b * dtheta)

    eb = math.exp(2.0 * math.pi * b)
    c_factor = (1.0 - p) + p * eb

    r0 = a
    r1 = a * math.exp(b * dtheta)
    rc0 = c_factor * r0
    rc1 = c_factor * r1

    p0 = _polar_to_cart(0.0, r0)
    p1 = _polar_to_cart(dtheta, r1)
    q0 = _polar_to_cart(0.0, rc0)
    q1 = _polar_to_cart(dtheta, rc1)

    dq = (q1[0] - q0[0], q1[1] - q0[1])
    angle = -math.atan2(dq[1], dq[0])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def rot(pt: Point2D) -> Point2D:
        x, y = pt
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    base = [rot(p0), rot(p1), rot(q1), rot(q0)]

    unit_count = max(1, unit_count)
    primary_polys: List[List[Point2D]] = []
    mirror_polys: List[List[Point2D]] = []

    current_x = 0.0
    for k in range(unit_count):
        scale = gamma**k
        scaled = [(x * scale, y * scale) for x, y in base]
        q0_scaled = scaled[3]
        q1_scaled = scaled[2]
        dx = current_x - q0_scaled[0]
        dy = -q0_scaled[1]
        placed = [(x + dx, y + dy) for x, y in scaled]
        placed_mirror = [(x, -y) for x, y in placed]
        primary_polys.append(placed)
        mirror_polys.append(placed_mirror)
        current_x = dx + q1_scaled[0]

    return primary_polys, mirror_polys


def _build_polar_units(
    a: float,
    b: float,
    dtheta_deg: int,
    turns: float,
    p: float,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[Tuple[List[float], List[float]]],
    List[Tuple[List[float], List[float]]],
    int,
]:
    theta_end = 2.0 * math.pi * turns
    rc_end = max(0.0, theta_end - 2.0 * math.pi)
    dtheta = math.radians(max(1, int(dtheta_deg)))

    theta_vals: List[float] = []
    r_vals: List[float] = []
    rc_vals: List[float] = []
    units_primary: List[Tuple[List[float], List[float]]] = []
    units_mirror: List[Tuple[List[float], List[float]]] = []

    eb = math.exp(2.0 * math.pi * b)
    c_factor = (1.0 - p) + p * eb

    theta = 0.0
    while theta <= theta_end + 1e-12:
        r = a * math.exp(b * theta)
        rc = c_factor * r
        theta_vals.append(theta)
        r_vals.append(r)
        rc_vals.append(rc)
        theta += dtheta

    unit_count = 0
    for i in range(len(theta_vals) - 1):
        t0 = theta_vals[i]
        t1 = theta_vals[i + 1]
        if t1 > rc_end + 1e-12:
            break
        r0 = r_vals[i]
        r1 = r_vals[i + 1]
        rc0 = rc_vals[i]
        rc1 = rc_vals[i + 1]
        units_primary.append(([t0, t1, t1, t0], [r0, r1, rc1, rc0]))
        # Mirror trapezoid across the central-spiral edge (q0-q1)
        p0 = _polar_to_cart(t0, r0)
        p1 = _polar_to_cart(t1, r1)
        q0 = _polar_to_cart(t0, rc0)
        q1 = _polar_to_cart(t1, rc1)
        p0m = _reflect_point_across_line(p0, q0, q1)
        p1m = _reflect_point_across_line(p1, q0, q1)
        t0m, r0m = _cart_to_polar(*p0m)
        t1m, r1m = _cart_to_polar(*p1m)
        units_mirror.append(([t0, t1, t1m, t0m], [rc0, rc1, r1m, r0m]))
        unit_count += 1

    return theta_vals, r_vals, rc_vals, units_primary, units_mirror, unit_count


def _polygons_to_polydata(polys: List[List[Point2D]]) -> vtkPolyData:
    points = vtkPoints()
    cells = vtkCellArray()
    for poly in polys:
        if len(poly) < 3:
            continue
        ids = []
        for x, y in poly:
            ids.append(points.InsertNextPoint(x, y, 0.0))
        poly_cell = vtkPolygon()
        poly_cell.GetPointIds().SetNumberOfIds(len(ids))
        for i, pid in enumerate(ids):
            poly_cell.GetPointIds().SetId(i, pid)
        cells.InsertNextCell(poly_cell)

    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def _extrude(polydata: vtkPolyData, thickness: float) -> vtkPolyData:
    extrude = vtkLinearExtrusionFilter()
    extrude.SetInputData(polydata)
    extrude.SetExtrusionTypeToVectorExtrusion()
    extrude.SetVector(0.0, 0.0, thickness)
    extrude.SetCapping(True)
    extrude.Update()
    # Center the extrusion around the XY plane (±thickness/2)
    transform = vtkTransform()
    transform.Translate(0.0, 0.0, -0.5 * thickness)
    tf = vtkTransformPolyDataFilter()
    tf.SetTransform(transform)
    tf.SetInputData(extrude.GetOutput())
    tf.Update()

    clean = vtkCleanPolyData()
    clean.SetInputData(tf.GetOutput())
    clean.Update()
    return clean.GetOutput()


# Distinct color identity per tendon (index 0 == tendon 1). Reused by the
# cross-section schematic so each cable is visually distinguishable.
TENDON_COLORS = [
    (0.90, 0.20, 0.20),  # 1 red
    (0.20, 0.60, 0.90),  # 2 blue
    (0.20, 0.72, 0.28),  # 3 green
    (0.95, 0.62, 0.10),  # 4 orange
    (0.62, 0.26, 0.85),  # 5 purple
    (0.10, 0.72, 0.70),  # 6 teal
    (0.88, 0.32, 0.68),  # 7 pink
    (0.45, 0.45, 0.45),  # 8 gray
]


def _tendon_color(idx: int) -> tuple[float, float, float]:
    return TENDON_COLORS[idx % len(TENDON_COLORS)]


class _SectionPreview(QWidget):
    """Schematic of the cross-section showing each tendon's angular position.

    Tendon k (1-based) is anchored at 90 + k*360/n degrees in the section plane
    (matches the site offsets in xml_generator.build_sites). The currently
    selected tendon is highlighted with a dark ring.
    """

    def __init__(self, num_cables: int, parent=None) -> None:
        super().__init__(parent)
        self._n = max(2, int(num_cables))
        self._current = 0
        self.setMinimumSize(150, 172)

    def set_current(self, idx: int) -> None:
        self._current = max(0, idx)
        self.update()

    def paintEvent(self, _event) -> None:
        from PySide6.QtGui import QBrush, QColor, QFont, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = int(w * 0.5), int(h * 0.52)
        radius = int(max(20, min(w, h) * 0.36))

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(0x55, 0x55, 0x55))
        p.drawText(6, 12, f"Cross-section ({self._n} cables)")

        # section disc
        p.setPen(QPen(QColor(0x40, 0x40, 0x40), 1.4))
        p.setBrush(QBrush(QColor(0xf2, 0xf4, 0xf7)))
        p.drawEllipse(cx - radius, cy - radius, 2 * radius, 2 * radius)

        # X / Y axes (Y is up on screen)
        p.setPen(QPen(QColor(0x90, 0x90, 0x90), 1, Qt.DashLine))
        p.drawLine(cx - radius - 4, cy, cx + radius + 4, cy)
        p.drawLine(cx, cy - radius - 4, cx, cy + radius + 4)
        p.setPen(QColor(0x70, 0x70, 0x70))
        p.drawText(cx + radius + 2, cy + 3, "X")
        p.drawText(cx + 2, cy - radius - 4, "Y")

        # tendon marks
        for k in range(self._n):
            theta = math.radians(90.0 + k * 360.0 / self._n)
            sx = cx + radius * math.cos(theta)
            sy = cy - radius * math.sin(theta)
            r, g, b = _tendon_color(k)
            qc = QColor(int(r * 255), int(g * 255), int(b * 255))

            if k == self._current:
                p.setPen(QPen(QColor(0x20, 0x20, 0x20), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(int(sx - 11), int(sy - 11), 22, 22)
                p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            else:
                p.setPen(QPen(QColor(0x20, 0x20, 0x20), 1))
                p.setFont(QFont("Segoe UI", 8))

            p.setBrush(QBrush(qc))
            p.drawEllipse(int(sx - 5), int(sy - 5), 10, 10)

            # number label pushed slightly along the outward direction
            lx = cx + (radius + 10) * math.cos(theta)
            ly = cy - (radius + 10) * math.sin(theta)
            p.setPen(QColor(0x20, 0x20, 0x20))
            p.drawText(int(lx - 4), int(ly + 3), str(k + 1))


class TendonConfigDialog(QDialog):
    """Per-tendon configuration dialog for MuJoCo XML export.

    Lets the user set each tendon's ctrl range, gear, and which joint the cable
    connects to. The joint is the "root -> that joint" semantic: a cable set to
    joint j spans links 1..j+1 and drives joints 1..j. "All (full chain)" means
    the cable runs from the root to the tip (links 1..N).
    """

    DEFAULT_GEAR = 50.0
    DEFAULT_CTRL_LO = -1.0
    DEFAULT_CTRL_HI = 1.0

    def __init__(self, num_tendons: int, num_joints: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tendon Configuration")
        self.num_tendons = max(2, int(num_tendons))
        self.num_joints = max(0, int(num_joints))
        # joint stored as int; num_joints means "All (full chain)";
        # 1..num_joints-1 -> root reaches that joint (short cable).
        self._default = {
            "gear": self.DEFAULT_GEAR,
            "ctrl_lo": self.DEFAULT_CTRL_LO,
            "ctrl_hi": self.DEFAULT_CTRL_HI,
            "joint": self.num_joints,
        }
        self._configs = [dict(self._default) for _ in range(self.num_tendons)]
        self._current = 0

        self._build_ui()
        self._load_current()
        self.section.set_current(self._current)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Tendon:"))
        self.tendon_combo = QComboBox()
        for i in range(self.num_tendons):
            self.tendon_combo.addItem(f"Tendon {i + 1}")
        sel_layout.addWidget(self.tendon_combo, 1)
        root_layout.addLayout(sel_layout)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        r = 0
        self.ctrl_lo_spin = QDoubleSpinBox()
        self._setup_spin(self.ctrl_lo_spin, -100.0, 0.0, 0.1, 2, self._default["ctrl_lo"])
        self.ctrl_hi_spin = QDoubleSpinBox()
        self._setup_spin(self.ctrl_hi_spin, 0.0, 100.0, 0.1, 2, self._default["ctrl_hi"])
        self.gear_spin = QDoubleSpinBox()
        self._setup_spin(self.gear_spin, 0.0, 10000.0, 1.0, 1, self._default["gear"])

        grid.addWidget(QLabel("Ctrl min"), r, 0)
        grid.addWidget(self.ctrl_lo_spin, r, 1)
        r += 1
        grid.addWidget(QLabel("Ctrl max"), r, 0)
        grid.addWidget(self.ctrl_hi_spin, r, 1)
        r += 1
        grid.addWidget(QLabel("Gear (N)"), r, 0)
        grid.addWidget(self.gear_spin, r, 1)
        r += 1
        grid.addWidget(QLabel("Connect to joint"), r, 0)
        self.joint_combo = QComboBox()
        # index 0 = All (root->tip); J1..J(num_joints-1) = root->that joint
        self.joint_combo.addItem("All (full chain)", self.num_joints)
        for j in range(1, self.num_joints):
            self.joint_combo.addItem(f"J{j}", j)
        grid.addWidget(self.joint_combo, r, 1)
        grid.setColumnStretch(1, 1)

        content = QHBoxLayout()
        content.addLayout(grid, 1)
        self.section = _SectionPreview(self.num_tendons)
        content.addWidget(self.section)
        root_layout.addLayout(content)

        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_current)
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)
        root_layout.addLayout(btn_layout)

        self.tendon_combo.currentIndexChanged.connect(self._on_combo_changed)

    def _setup_spin(self, spin, vmin, vmax, step, decimals, value) -> None:
        spin.setRange(vmin, vmax)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        spin.setValue(value)
        spin.setFixedWidth(120)

    def _save_current(self) -> None:
        self._configs[self._current] = {
            "gear": self.gear_spin.value(),
            "ctrl_lo": self.ctrl_lo_spin.value(),
            "ctrl_hi": self.ctrl_hi_spin.value(),
            "joint": self.joint_combo.currentData(),
        }

    def _load_current(self) -> None:
        cfg = self._configs[self._current]
        widgets = (self.gear_spin, self.ctrl_lo_spin, self.ctrl_hi_spin, self.joint_combo)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.gear_spin.setValue(cfg["gear"])
            self.ctrl_lo_spin.setValue(cfg["ctrl_lo"])
            self.ctrl_hi_spin.setValue(cfg["ctrl_hi"])
            idx = 0
            target = cfg["joint"]
            if target < self.num_joints:
                for i in range(self.joint_combo.count()):
                    if self.joint_combo.itemData(i) == target:
                        idx = i
                        break
            self.joint_combo.setCurrentIndex(idx)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _on_combo_changed(self, _idx: int) -> None:
        self._save_current()
        self._current = self.tendon_combo.currentIndex()
        self._load_current()
        self.section.set_current(self._current)

    def _reset_current(self) -> None:
        self._configs[self._current] = dict(self._default)
        self._load_current()

    def _on_ok(self) -> None:
        self._save_current()
        self.accept()

    def get_configs(self) -> list[dict]:
        return [dict(c) for c in self._configs]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpiRobs Design Tool")
        self.params = Params()
        self._polys_primary: List[List[Point2D]] = []
        self._polys_mirror: List[List[Point2D]] = []
        self._elastic_poly: List[Point2D] | None = None
        self._elastic_poly_mirror: List[Point2D] | None = None
        self._ray_start: Point2D | None = None
        self._ray_upper_end: Point2D | None = None
        self._ray_lower_end: Point2D | None = None
        self._has_3d = False
        self._show_cone2_preview = False
        self._vtk_initialized = False
        # showEvent fires on every taskbar window switch (Windows re-sends it on
        # re-display). Rebuilding the whole VTK / CadQuery pipeline each time
        # leaves stale actors that the renderer still references → access
        # violation (UAF). Only run the heavy init once; later shows just ensure
        # VTK is ready and let Qt repaint.
        self._show_initialized = False
        self._camera_initialized = False
        self._extrusion_initialized = False
        self._axes_actor = vtkAxesActor()
        self._axes_actor.SetTotalLength(20.0, 20.0, 20.0)
        self._axes_actor.SetShaftTypeToCylinder()
        self._axes_actor.SetCylinderRadius(0.02)
        self._axes_actor.SetAxisLabels(False)
        # Corner legend: color -> axis (matches vtkAxesActor default colors:
        # X = red, Y = green, Z = blue). Fixed at the lower-left of the viewport.
        self._axes_legend = vtkLegendBoxActor()
        self._axes_legend.SetNumberOfEntries(3)
        self._axes_legend.SetEntryColor(0, 1.0, 0.0, 0.0)
        self._axes_legend.SetEntryString(0, "X")
        self._axes_legend.SetEntryColor(1, 0.0, 1.0, 0.0)
        self._axes_legend.SetEntryString(1, "Y")
        self._axes_legend.SetEntryColor(2, 0.0, 0.0, 1.0)
        self._axes_legend.SetEntryString(2, "Z")
        self._axes_legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self._axes_legend.GetPositionCoordinate().SetValue(0.02, 0.02, 0.0)
        self._axes_legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        self._axes_legend.GetPosition2Coordinate().SetValue(0.10, 0.14, 0.0)
        self._axes_legend.SetPadding(5)
        self._axes_legend.UseBackgroundOff()
        _legend_tp = self._axes_legend.GetEntryTextProperty()
        _legend_tp.SetColor(0.15, 0.15, 0.15)
        _legend_tp.SetFontSize(12)
        self._base_size = 0.0
        self._tip_size = 0.0
        self._taper_angle_deg = 0.0
        self._robot_length = 0.0
        self._last_polar = None
        self._last_cart = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # Left: 2D plots
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(6)
        left_title = QLabel("2D Sketch")
        left_title.setAlignment(Qt.AlignHCenter)
        left_title.setStyleSheet("font-weight:600; color:#555;")
        left_panel_layout.addWidget(left_title)
        left_split = QSplitter(Qt.Vertical)
        left_split.setOpaqueResize(False)
        left_split.setChildrenCollapsible(False)

        self.polar_fig = Figure(figsize=(5, 4))
        self.polar_canvas = FigureCanvas(self.polar_fig)
        self.polar_ax = self.polar_fig.add_subplot(111, projection="polar")
        left_split.addWidget(self.polar_canvas)

        self.cart_fig = Figure(figsize=(5, 4))
        self.cart_canvas = FigureCanvas(self.cart_fig)
        self.cart_ax = self.cart_fig.add_subplot(111)
        left_split.addWidget(self.cart_canvas)

        # Outputs below cartesian plot
        info_panel = QWidget()
        info_panel.setStyleSheet("background-color: #ffffff;")
        info_layout = QGridLayout(info_panel)
        info_layout.setContentsMargins(12, 6, 12, 6)
        info_layout.setHorizontalSpacing(30)
        info_layout.setVerticalSpacing(6)
        self.taper_label = QLabel("Taper Angle: --")
        self.tip_label = QLabel("Tip Size: --")
        self.base_label = QLabel("Base Size: --")
        self.length_label = QLabel("Robot Length: --")
        self.units_label = QLabel("Num of Units: --")
        for lbl in (
            self.taper_label,
            self.tip_label,
            self.base_label,
            self.length_label,
            self.units_label,
        ):
            lbl.setStyleSheet("color: #7a7f87; font-size: 14px;")
        info_layout.addWidget(self.taper_label, 0, 0)
        info_layout.addWidget(self.tip_label, 0, 1)
        info_layout.addWidget(self.units_label, 0, 2)
        info_layout.addWidget(self.length_label, 1, 0)
        info_layout.addWidget(self.base_label, 1, 1)
        left_split.addWidget(info_panel)
        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 2)
        left_split.setStretchFactor(2, 1)
        left_split.setSizes([460, 300, 90])
        left_panel_layout.addWidget(left_split, 1)

        # VTK viewport (3D) with placeholder overlay
        if QVTKOpenGLNativeWidget is not None:
            self.vtk_widget = QVTKOpenGLNativeWidget()
        else:
            self.vtk_widget = QVTKRenderWindowInteractor()
        if sys.platform == "darwin":
            # Qt 6.10 on macOS otherwise retriggers paintEvent from each VTK
            # render, causing an infinite render/repaint loop.
            self.vtk_widget.setAttribute(Qt.WA_PaintOnScreen, False)
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtk_widget.setMinimumSize(240, 200)
        self.vtk_placeholder = QLabel("3D Preview (placeholder)")
        self.vtk_placeholder.setAlignment(Qt.AlignCenter)
        self.vtk_placeholder.setStyleSheet(
            "background-color: #f5f6f8; border: 1px dashed #d0d5dc;"
            " color: #9aa0a6; font-size: 14px;"
        )
        self.vtk_placeholder.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        vtk_container = QWidget()
        vtk_layout = QGridLayout(vtk_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_layout.addWidget(self.vtk_widget, 0, 0)
        vtk_layout.addWidget(self.vtk_placeholder, 0, 0)

        vtk_panel = QWidget()
        vtk_panel_layout = QVBoxLayout(vtk_panel)
        vtk_panel_layout.setContentsMargins(0, 0, 0, 0)
        vtk_panel_layout.setSpacing(6)
        vtk_title = QLabel("3D Model")
        vtk_title.setAlignment(Qt.AlignHCenter)
        vtk_title.setStyleSheet("font-weight:600; color:#555;")
        vtk_panel_layout.addWidget(vtk_title)
        vtk_panel_layout.addWidget(vtk_container, 1)
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().SetMultiSamples(0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.96, 0.97, 0.98)

        # Controls
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setAlignment(Qt.AlignTop)
        panel_layout.setSpacing(10)

        self._param_label_width = 120
        self._param_slider_width = 140
        self._param_spin_width = 90

        form_2d = QGridLayout()
        form_2d.setContentsMargins(0, 0, 0, 0)
        form_2d.setHorizontalSpacing(8)
        row = 0
        self.a_spin, self.a_slider, row = self._add_double_control(
            form_2d, row, "a", "(mm)", 0.1, 100.0, 0.01, 4, self.params.a, scale=1000
        )
        self.b_spin, self.b_slider, row = self._add_double_control(
            form_2d, row, "b", "", 0.01, 0.35, 0.001, 4, self.params.b, scale=10000
        )
        self.dtheta_spin, self.dtheta_slider, row = self._add_int_control(
            form_2d, row, "Δθ", "(deg)", 1, 60, self.params.dtheta_deg
        )
        self.theta_max_spin, self.theta_max_slider, row = self._add_double_control(
            form_2d, row, "θ max", "(π)", 1.0, 12.0, 0.1, 1, self.params.theta_max_pi, scale=10
        )
        self.p_spin, self.p_slider, row = self._add_double_control(
            form_2d, row, "p", "", 0.0, 0.5, 0.01, 2, self.params.p, scale=100
        )

        label_2d = QLabel("2D Parameters")
        label_2d.setStyleSheet("font-weight:600; color:#666;")
        panel_layout.addWidget(label_2d)
        panel_layout.addLayout(form_2d)

        self.save_img_btn = QPushButton("Save 2D Sketch")
        panel_layout.addWidget(self.save_img_btn)

        label_holes = QLabel("Fabrication Parameters")
        label_holes.setStyleSheet("font-weight:600; color:#666;")
        panel_layout.addWidget(label_holes)
        form_3d = QGridLayout()
        form_3d.setContentsMargins(0, 0, 0, 0)
        form_3d.setHorizontalSpacing(8)

        row = 0
        elastic_row = QHBoxLayout()
        elastic_row.setContentsMargins(0, 0, 0, 0)
        elastic_row.setSpacing(8)
        self.elastic_check = ToggleSwitch(self.params.elastic_enabled)
        elastic_label = QLabel("Elastic Layer/Axis")
        elastic_row.addWidget(self.elastic_check)
        elastic_row.addWidget(elastic_label)
        elastic_row.addStretch(1)
        elastic_wrap = QWidget()
        elastic_wrap.setLayout(elastic_row)
        form_3d.addWidget(elastic_wrap, row, 0, 1, 3)
        row += 1

        self.elastic_spin, self.elastic_slider, row = self._add_double_control(
            form_3d, row, "Elastic", "(%)", 0.0, 100.0, 1.0, 1, self.params.elastic_percent, scale=10
        )

        self.tip_hole_pos_spin, self.tip_hole_pos_slider, row = self._add_double_control(
            form_3d,
            row,
            "Pos_TipHole",
            "(%)",
            0.0,
            100.0,
            1.0,
            0,
            self.params.tip_hole_pos,
            scale=1,
        )
        self.tip_hole_size_spin, self.tip_hole_size_slider, row = self._add_double_control(
            form_3d,
            row,
            "Size_TipHole",
            "(mm)",
            0.1,
            10.0,
            0.1,
            2,
            self.params.tip_hole_size,
            scale=10,
        )
        self.base_hole_pos_spin, self.base_hole_pos_slider, row = self._add_double_control(
            form_3d,
            row,
            "Pos_BaseHole",
            "(%)",
            0.0,
            100.0,
            1.0,
            0,
            self.params.base_hole_pos,
            scale=1,
        )
        self.base_hole_size_spin, self.base_hole_size_slider, row = self._add_double_control(
            form_3d,
            row,
            "Size_BaseHole",
            "(mm)",
            0.1,
            10.0,
            0.1,
            2,
            self.params.base_hole_size,
            scale=10,
        )

        teardrop_row = QHBoxLayout()
        teardrop_row.setContentsMargins(0, 0, 0, 0)
        teardrop_row.setSpacing(8)
        self.teardrop_check = ToggleSwitch(self.params.teardrop_holes)
        teardrop_label = QLabel("Teardrop Holes (+Z)")
        teardrop_row.addWidget(self.teardrop_check)
        teardrop_row.addWidget(teardrop_label)
        teardrop_row.addStretch(1)
        teardrop_wrap = QWidget()
        teardrop_wrap.setLayout(teardrop_row)
        form_3d.addWidget(teardrop_wrap, row, 0, 1, 3)
        row += 1

        cable_row = QHBoxLayout()
        cable_row.setContentsMargins(0, 0, 0, 0)
        cable_row.setSpacing(8)
        self.cable_mode_title = QLabel("Num of Cables")
        self.cable2_check = QCheckBox("2")
        self.cable3_check = QCheckBox("N")
        self.cable2_check.setChecked(self.params.two_cable)
        self.cable3_check.setChecked(not self.params.two_cable)
        self.cable_count_spin = QSpinBox()
        self.cable_count_spin.setRange(3, 8)
        self.cable_count_spin.setSingleStep(1)
        self.cable_count_spin.setKeyboardTracking(False)
        self.cable_count_spin.setValue(self.params.num_cables)
        self.cable_count_spin.setFixedWidth(60)
        cable_row.addWidget(self.cable_mode_title)
        cable_row.addWidget(self.cable2_check)
        cable_row.addWidget(self.cable3_check)
        cable_row.addWidget(self.cable_count_spin)
        cable_row.addStretch(1)
        cable_wrap = QWidget()
        cable_wrap.setLayout(cable_row)
        form_3d.addWidget(cable_wrap, row, 0, 1, 3)
        row += 1

        self.cable2_wrap = QWidget()
        cable2_layout = QGridLayout(self.cable2_wrap)
        cable2_layout.setContentsMargins(0, 0, 0, 0)
        cable2_layout.setHorizontalSpacing(8)
        cable_row_idx = 0
        self.extrusion_spin, self.extrusion_slider, cable_row_idx = self._add_double_control(
            cable2_layout,
            cable_row_idx,
            "Extrusion",
            "(mm)",
            0.1,
            200.0,
            0.1,
            2,
            self.params.extrusion,
            scale=10,
        )
        self.cone1_spin, self.cone1_slider, cable_row_idx = self._add_double_control(
            cable2_layout,
            cable_row_idx,
            "ConeAngle1",
            "(deg)",
            0.0,
            45.0,
            0.5,
            1,
            self.params.cone_angle1,
            scale=10,
        )
        self.cone2_spin, self.cone2_slider, cable_row_idx = self._add_double_control(
            cable2_layout,
            cable_row_idx,
            "ConeAngle2",
            "(deg)",
            -45.0,
            45.0,
            0.5,
            1,
            self.params.cone_angle2,
            scale=10,
        )
        self.show_cone_cuts_check = QCheckBox("Show Cone Cuts (yellow)")
        self.show_cone_cuts_check.setChecked(False)
        self.show_cone_cuts_check.toggled.connect(self._on_show_cone_cuts_toggled)
        cable2_layout.addWidget(self.show_cone_cuts_check, cable_row_idx, 0, 1, 3)
        cable_row_idx += 1
        form_3d.addWidget(self.cable2_wrap, row, 0, 1, 3)
        row += 1

        self.cable3_wrap = QWidget()
        cable3_layout = QGridLayout(self.cable3_wrap)
        cable3_layout.setContentsMargins(0, 0, 0, 0)
        cable3_layout.setHorizontalSpacing(8)
        cable3_row_idx = 0
        self.cable3_cut_check = QCheckBox("Enable Edge Cut")
        self.cable3_cut_check.setChecked(self.params.cable3_cut_enabled)
        cable3_layout.addWidget(self.cable3_cut_check, cable3_row_idx, 0, 1, 3)
        cable3_row_idx += 1

        self.cable3_cut_params_wrap = QWidget()
        cable3_cut_params_layout = QGridLayout(self.cable3_cut_params_wrap)
        cable3_cut_params_layout.setContentsMargins(0, 0, 0, 0)
        cable3_cut_params_layout.setHorizontalSpacing(8)
        cable3_cut_row = 0
        self.cable3_cut_pos_spin, self.cable3_cut_pos_slider, cable3_cut_row = self._add_double_control(
            cable3_cut_params_layout,
            cable3_cut_row,
            "Cut Start (%)",
            "",
            0.0,
            50.0,
            1.0,
            1,
            self.params.cable3_cut_pos,
            scale=10,
        )
        self.cable3_cut_size_spin, self.cable3_cut_size_slider, cable3_cut_row = self._add_double_control(
            cable3_cut_params_layout,
            cable3_cut_row,
            "Cut Size (%)",
            "",
            0.0,
            20.0,
            0.1,
            2,
            self.params.cable3_cut_size,
            scale=10,
        )
        cable3_layout.addWidget(self.cable3_cut_params_wrap, cable3_row_idx, 0, 1, 3)
        cable3_row_idx += 1

        form_3d.addWidget(self.cable3_wrap, row, 0, 1, 3)
        row += 1

        panel_layout.addLayout(form_3d)

        label_sim = QLabel("Simulation Parameters")
        label_sim.setStyleSheet("font-weight:600; color:#666;")
        panel_layout.addWidget(label_sim)

        form_sim = QGridLayout()
        form_sim.setContentsMargins(0, 0, 0, 0)
        form_sim.setHorizontalSpacing(8)
        sim_row = 0
        self.sim_stiffness_spin, self.sim_stiffness_slider, sim_row = self._add_double_control(
            form_sim,
            sim_row,
            "Stiffness",
            "",
            0.0,
            100.0,
            0.1,
            3,
            self.params.sim_stiffness,
            scale=10,
        )
        self.sim_damping_spin, self.sim_damping_slider, sim_row = self._add_double_control(
            form_sim,
            sim_row,
            "Damping",
            "",
            0.0,
            100.0,
            0.1,
            3,
            self.params.sim_damping,
            scale=10,
        )
        self.sim_gravity_spin, self.sim_gravity_slider, sim_row = self._add_double_control(
            form_sim,
            sim_row,
            "Gravity",
            "m/s^2",
            0.0,
            30.0,
            0.1,
            2,
            self.params.gravity,
            scale=10,
        )
        panel_layout.addLayout(form_sim)

        # Joint pose is an optional feature: the checkbox lives OUTSIDE the
        # container so it stays reachable while the section is collapsed.
        self.pose_check = QCheckBox("Enable Joint Pose")
        self.pose_check.setChecked(self.params.pose_enabled)
        panel_layout.addWidget(self.pose_check)

        self.pose_wrap = QWidget()
        pose_layout = QVBoxLayout(self.pose_wrap)
        pose_layout.setContentsMargins(0, 0, 0, 0)

        label_pose = QLabel("Joint Pose")
        label_pose.setStyleSheet("font-weight:600; color:#666;")
        pose_layout.addWidget(label_pose)

        form_pose = QGridLayout()
        form_pose.setContentsMargins(0, 0, 0, 0)
        form_pose.setHorizontalSpacing(8)
        pose_row = 0
        self.bend_dir_spin, self.bend_dir_slider, pose_row = self._add_double_control(
            form_pose,
            pose_row,
            "Bend Dir",
            "(deg)",
            0.0,
            360.0,
            1.0,
            0,
            self.params.bend_dir_deg,
            scale=10,
        )
        self.bend_dir_wrap = QWidget()
        self.bend_dir_wrap.setLayout(form_pose)
        pose_layout.addWidget(self.bend_dir_wrap)

        self.joint_wrap = QWidget()
        joint_layout = QVBoxLayout(self.joint_wrap)
        joint_layout.setContentsMargins(0, 0, 0, 0)
        self.filler_auto_check = QCheckBox("Auto filler size")
        self.filler_auto_check.setChecked(self.params.filler_auto)
        joint_layout.addWidget(self.filler_auto_check)
        self.joint_selector = QComboBox()
        joint_layout.addWidget(self.joint_selector)
        self.joint_stack = QStackedWidget()
        joint_layout.addWidget(self.joint_stack)
        self.joint_selector.currentIndexChanged.connect(self.joint_stack.setCurrentIndex)
        self._joint_spins = []
        self._joint_filler_widths = []
        self._joint_filler_heights = []
        pose_layout.addWidget(self.joint_wrap)

        self.reset_pose_btn = QPushButton("Reset Pose")
        pose_layout.addWidget(self.reset_pose_btn)

        panel_layout.addWidget(self.pose_wrap)

        label_actions = QLabel("Actions")
        label_actions.setStyleSheet("font-weight:600; color:#666;")

        panel_layout.addStretch(1)

        panel_layout.addWidget(label_actions)
        self.update_btn = QPushButton("Generate 3D Model")
        panel_layout.addWidget(self.update_btn)
        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.setStyleSheet("QPushButton { background-color: #f0f4e6; } QPushButton:hover { background-color: #e6eed6; }")
        panel_layout.addWidget(self.save_params_btn)
        self.reset_params_btn = QPushButton("Reset Parameters")
        panel_layout.addWidget(self.reset_params_btn)
        self.reset_view_btn = QPushButton("Reset 3D View")
        panel_layout.addWidget(self.reset_view_btn)

        self.export_cad_btn = QPushButton("Export STEP/STL")
        self.export_cad_btn.setStyleSheet("QPushButton { background-color: #e6f0ff; } QPushButton:hover { background-color: #d7e6ff; }")
        panel_layout.addWidget(self.export_cad_btn)
        self.export_xml_btn = QPushButton("Export XML")
        self.export_xml_btn.setStyleSheet("QPushButton { background-color: #e6f0ff; } QPushButton:hover { background-color: #d7e6ff; }")
        panel_layout.addWidget(self.export_xml_btn)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)

        panel_title = QLabel("Parameters")
        panel_title.setAlignment(Qt.AlignHCenter)
        panel_title.setStyleSheet("font-weight:600; color:#555;")
        panel_wrap = QWidget()
        panel_wrap_layout = QVBoxLayout(panel_wrap)
        panel_wrap_layout.setContentsMargins(0, 0, 0, 0)
        panel_wrap_layout.setSpacing(6)
        panel_wrap_layout.addWidget(panel_title)
        panel_wrap_layout.addWidget(scroll, 1)

        self.main_split = QSplitter(Qt.Horizontal)
        self.main_split.setOpaqueResize(False)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.addWidget(left_panel)
        self.main_split.addWidget(vtk_panel)
        self.main_split.addWidget(panel_wrap)
        self.main_split.setStretchFactor(0, 4)
        self.main_split.setStretchFactor(1, 4)
        self.main_split.setStretchFactor(2, 2)
        layout.addWidget(self.main_split, 1)

        self._splitter_timer = QTimer(self)
        self._splitter_timer.setSingleShot(True)
        self._splitter_timer.timeout.connect(self._splitter_idle)
        self.main_split.splitterMoved.connect(self._on_splitter_moved)
        left_split.splitterMoved.connect(self._on_splitter_moved)

        self.update_btn.clicked.connect(self.update_scene)
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_2d)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._finish_resize)
        self._resizing = False
        self.save_img_btn.clicked.connect(self.save_image)
        self.save_params_btn.clicked.connect(self._save_params)
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        self.reset_pose_btn.clicked.connect(self._reset_pose)
        self.export_cad_btn.clicked.connect(self.export_cad)
        self.export_xml_btn.clicked.connect(self.export_xml)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.cable2_check.toggled.connect(self._on_cable2_checked)
        self.cable3_check.toggled.connect(self._on_cable3_checked)
        self.cable3_cut_check.toggled.connect(self._on_cable3_cut_toggled)
        self.pose_check.toggled.connect(self._on_pose_toggled)
        self.filler_auto_check.toggled.connect(self._on_filler_auto_toggled)
        self._on_cable_toggle(self.cable2_check.isChecked())
        for w in (
            self.a_spin,
            self.b_spin,
            self.dtheta_spin,
            self.theta_max_spin,
            self.p_spin,
            self.elastic_spin,
            self.extrusion_spin,
            self.cone1_spin,
            self.cone2_spin,
            self.cable3_cut_pos_spin,
            self.cable3_cut_size_spin,
            self.tip_hole_pos_spin,
            self.tip_hole_size_spin,
            self.base_hole_pos_spin,
            self.base_hole_size_spin,
            self.a_slider,
            self.b_slider,
            self.dtheta_slider,
            self.theta_max_slider,
            self.p_slider,
            self.elastic_slider,
            self.extrusion_slider,
            self.cone1_slider,
            self.cone2_slider,
            self.cable3_cut_pos_slider,
            self.cable3_cut_size_slider,
            self.tip_hole_pos_slider,
            self.tip_hole_size_slider,
            self.base_hole_pos_slider,
            self.base_hole_size_slider,
            self.cable_count_spin,
        ):
            w.valueChanged.connect(self.schedule_update)
        self.elastic_check.toggled.connect(lambda _v: self.schedule_update())
        self.teardrop_check.toggled.connect(lambda _v: self.schedule_update())

        self._apply_default_parameters(update_scene=False)
        self.pose_wrap.setVisible(self.pose_check.isChecked())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Only run the heavy rebuild-once init on the first real show. On later
        # shows (taskbar window switch, minimize/restore) do NOT reload params or
        # rebuild the VTK pipeline — that re-creates actors the renderer already
        # references, freeing them out from under OpenCascade/VTK → access
        # violation. Just make sure VTK is up; Qt repaints it on its own.
        if self._show_initialized:
            self._ensure_vtk_ready()
            return
        self._show_initialized = True
        QTimer.singleShot(0, self._apply_splitter_sizes)
        # Build the model + apply saved params AFTER the first paint. The OpenGL/
        # VTK init and OCCT model build ran synchronously here before, blocking the
        # window from rendering and making startup feel very slow. Deferring them
        # lets the window and all controls appear immediately; the 3D model then
        # fills in a moment later.
        QTimer.singleShot(0, self._startup_build)

    def _startup_build(self) -> None:
        self._ensure_vtk_ready()
        self._load_and_apply_saved_params()
        self.update_scene()

    def _apply_splitter_sizes(self) -> None:
        if not hasattr(self, "main_split"):
            return
        total = max(1, self.main_split.width())
        left = int(total * 0.4)
        mid = int(total * 0.4)
        right = max(1, total - left - mid)
        self.main_split.setSizes([left, mid, right])

    def _ensure_vtk_ready(self) -> None:
        if self._vtk_initialized:
            return
        try:
            self.vtk_widget.Initialize()
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            if interactor is not None:
                interactor.Initialize()
                # NOTE: Do NOT call interactor.Start() — it runs a continuous
                # render loop that conflicts with Qt's event loop and causes lag.
                # QVTKOpenGLNativeWidget handles rendering via Qt paint events.
                interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            self._vtk_initialized = True
            self.vtk_placeholder.setVisible(False)
        except Exception as exc:
            self._vtk_initialized = False
            self.vtk_placeholder.setText(f"3D Preview unavailable: {exc}")
            self.vtk_placeholder.setVisible(True)

    def schedule_update(self) -> None:
        if self._resizing:
            return
        self._update_timer.start(200)
        # Only update cone ranges when their inputs actually changed
        ext = float(self.extrusion_spin.value())
        rlen = self._robot_length
        if (getattr(self, "_cached_cone1_ext", None) != ext
                or getattr(self, "_cached_cone1_len", None) != rlen):
            self._cached_cone1_ext = ext
            self._cached_cone1_len = rlen
            self._update_cone1_range()
        taper = self._taper_angle_deg
        if getattr(self, "_cached_cone2_taper", None) != taper:
            self._cached_cone2_taper = taper
            self._update_cone2_range()

    def _on_cable2_checked(self, checked: bool) -> None:
        if checked:
            if self.cable3_check.isChecked():
                self.cable3_check.blockSignals(True)
                self.cable3_check.setChecked(False)
                self.cable3_check.blockSignals(False)
            self._on_cable_toggle(True)
        elif not self.cable3_check.isChecked():
            self.cable2_check.blockSignals(True)
            self.cable2_check.setChecked(True)
            self.cable2_check.blockSignals(False)

    def _on_cable3_checked(self, checked: bool) -> None:
        if checked:
            if self.cable2_check.isChecked():
                self.cable2_check.blockSignals(True)
                self.cable2_check.setChecked(False)
                self.cable2_check.blockSignals(False)
            self._on_cable_toggle(False)
        elif not self.cable2_check.isChecked():
            self.cable3_check.blockSignals(True)
            self.cable3_check.setChecked(True)
            self.cable3_check.blockSignals(False)

    def _on_cable3_cut_toggled(self, checked: bool) -> None:
        self.params.cable3_cut_enabled = bool(checked)
        self.cable3_cut_params_wrap.setVisible((not self.params.two_cable) and bool(checked))
        self.schedule_update()

    def _on_pose_toggled(self, checked: bool) -> None:
        self.params.pose_enabled = bool(checked)
        self.pose_wrap.setVisible(bool(checked))
        if not checked:
            # Disabling the feature cancels the bend: the model goes straight.
            for s in getattr(self, "_joint_spins", []):
                s.setValue(0.0)
        self.update_scene()

    def _on_filler_auto_toggled(self, checked: bool) -> None:
        self.params.filler_auto = bool(checked)
        if checked:
            self._sync_auto_filler_sizes()
            self.update_scene()

    def _sync_auto_filler_sizes(self) -> None:
        """Push the auto-derived filler sizes back into the per-joint boxes.

        The boxes are not wired to schedule_update, so setValue() cannot
        recurse back into this method.
        """
        if not getattr(self, "_joint_filler_widths", None):
            return
        widths, heights = self._joint_filler_defaults()
        for box, v in zip(self._joint_filler_widths, widths):
            box.setValue(v)
        for box, v in zip(self._joint_filler_heights, heights):
            box.setValue(v)

    def _on_show_cone_cuts_toggled(self, _checked: bool) -> None:
        # Only meaningful for the 2-cable model (cone cuts don't exist in 3-cable).
        # schedule_update refreshes the 2D plots but NOT the 3D preview. Flipping
        # this checkbox must (re)draw the yellow cut region, so rebuild the scene
        # directly. In 3-cable mode the toggle has no visible effect and we let the
        # 2D refresh stand.
        if self.params.two_cable:
            self.update_scene()
        else:
            self.schedule_update()

    def _on_cable_toggle(self, checked: bool) -> None:
        self.params.two_cable = bool(checked)
        self.cable2_wrap.setVisible(checked)
        self.cable3_wrap.setVisible(not checked)
        self.cable_count_spin.setEnabled(not checked)
        self.cable3_cut_params_wrap.setVisible((not checked) and self.cable3_cut_check.isChecked())
        if hasattr(self, "bend_dir_wrap"):
            self.bend_dir_wrap.setVisible(not checked)
        self.update_scene()

    def _cable_angles(self) -> List[float]:
        n = 2 if self.params.two_cable else max(3, int(self.params.num_cables))
        return [k * 360.0 / n for k in range(n)]

    def _on_splitter_moved(self, *_args) -> None:
        # Pause heavy redraw during drag to avoid ghosting
        self.vtk_widget.setUpdatesEnabled(False)
        self._splitter_timer.start(120)

    def _splitter_idle(self) -> None:
        self.vtk_widget.setUpdatesEnabled(True)
        self.vtk_widget.GetRenderWindow().Render()
        self.polar_canvas.draw_idle()
        self.cart_canvas.draw_idle()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resizing = True
        # Do NOT disable the matplotlib canvases: suppressing their paint events
        # leaves the 2D plots/panel black until the delayed finish_resize runs,
        # while the VTK OpenGL view keeps rendering independently (so only the
        # non-3D areas turn black). Their draw_idle() already throttles redraws.
        self.vtk_widget.setUpdatesEnabled(False)
        self._resize_timer.start(120)

    def _finish_resize(self) -> None:
        self.vtk_widget.setUpdatesEnabled(True)
        self.vtk_widget.GetRenderWindow().Render()
        self.polar_canvas.draw_idle()
        self.cart_canvas.draw_idle()
        self._resizing = False

    def _add_double_control(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        unit: str,
        vmin: float,
        vmax: float,
        step: float,
        decimals: int,
        value: float,
        scale: int,
    ) -> Tuple[QDoubleSpinBox, QSlider, int]:
        box = QDoubleSpinBox()
        box.setRange(vmin, vmax)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setKeyboardTracking(False)
        box.setValue(value)
        box.setFixedWidth(self._param_spin_width)

        slider = QSlider(Qt.Horizontal)
        slider.setFixedWidth(self._param_slider_width)
        slider.setRange(int(vmin * scale), int(vmax * scale))
        slider.setSingleStep(max(1, int(step * scale)))
        slider.setValue(int(value * scale))

        box.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))
        slider.valueChanged.connect(lambda v: box.setValue(v / scale))

        label_text = label if not unit else f"{label} {unit}"
        label_widget = QLabel(label_text)
        label_widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        label_widget.setFixedWidth(self._param_label_width)

        grid.addWidget(label_widget, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(box, row, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, self._param_spin_width)
        return box, slider, row + 1

    def _add_int_control(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        unit: str,
        vmin: int,
        vmax: int,
        value: int,
    ) -> Tuple[QSpinBox, QSlider, int]:
        box = QSpinBox()
        box.setRange(vmin, vmax)
        box.setKeyboardTracking(False)
        box.setValue(value)
        box.setFixedWidth(self._param_spin_width)

        slider = QSlider(Qt.Horizontal)
        slider.setFixedWidth(self._param_slider_width)
        slider.setRange(vmin, vmax)
        slider.setSingleStep(1)
        slider.setValue(value)

        box.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(box.setValue)

        label_text = label if not unit else f"{label} {unit}"
        label_widget = QLabel(label_text)
        label_widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        label_widget.setFixedWidth(self._param_label_width)

        grid.addWidget(label_widget, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(box, row, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, self._param_spin_width)
        return box, slider, row + 1

        
    def _params_file_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "exports", "last_params.json")

    def _save_params(self) -> None:
        """Save current parameters to a JSON file."""
        import json

        out_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(out_dir, exist_ok=True)
        data = {
            "a": float(self.a_spin.value()),
            "b": float(self.b_spin.value()),
            "dtheta_deg": int(self.dtheta_spin.value()),
            "theta_max_pi": float(self.theta_max_spin.value()),
            "p": float(self.p_spin.value()),
            "elastic_percent": float(self.elastic_spin.value()),
            "elastic_enabled": self.elastic_check.isChecked(),
            "extrusion": float(self.extrusion_spin.value()),
            "cone_angle1": float(self.cone1_spin.value()),
            "cone_angle2": float(self.cone2_spin.value()),
            "show_cone_cuts": self.show_cone_cuts_check.isChecked(),
            "tip_hole_pos": float(self.tip_hole_pos_spin.value()),
            "tip_hole_size": float(self.tip_hole_size_spin.value()),
            "base_hole_pos": float(self.base_hole_pos_spin.value()),
            "base_hole_size": float(self.base_hole_size_spin.value()),
            "teardrop_holes": self.teardrop_check.isChecked(),
            "two_cable": self.cable2_check.isChecked(),
            "num_cables": int(self.cable_count_spin.value()),
            "cable3_cut_enabled": self.cable3_cut_check.isChecked(),
            "cable3_cut_pos": float(self.cable3_cut_pos_spin.value()),
            "cable3_cut_size": float(self.cable3_cut_size_spin.value()),
            "sim_stiffness": float(self.sim_stiffness_spin.value()),
            "sim_damping": float(self.sim_damping_spin.value()),
            "sim_gravity": float(self.sim_gravity_spin.value()),
            "pose_enabled": self.pose_check.isChecked(),
            "filler_auto": self.filler_auto_check.isChecked(),
            "bend_dir_deg": float(self.bend_dir_spin.value()),
            "joint_angles": [float(s.value()) for s in getattr(self, "_joint_spins", [])],
            "joint_filler_widths": [float(s.value()) for s in getattr(self, "_joint_filler_widths", [])],
            "joint_filler_heights": [float(s.value()) for s in getattr(self, "_joint_filler_heights", [])],
        }
        with open(self._params_file_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_and_apply_saved_params(self) -> bool:
        """Load saved parameters from JSON and apply to widgets.

        Returns True if saved parameters were loaded successfully.
        """
        import json

        params_file = self._params_file_path()
        if not os.path.isfile(params_file):
            return False
        try:
            with open(params_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False

        # --- 2D geometry params ---
        if "a" in data:
            self.a_spin.setValue(float(data["a"]))
        if "b" in data:
            self.b_spin.setValue(float(data["b"]))
        if "dtheta_deg" in data:
            self.dtheta_spin.setValue(int(data["dtheta_deg"]))
        if "theta_max_pi" in data:
            self.theta_max_spin.setValue(float(data["theta_max_pi"]))
        if "p" in data:
            self.p_spin.setValue(float(data["p"]))

        # --- Elastic layer ---
        if "elastic_enabled" in data:
            self.elastic_check.setChecked(bool(data["elastic_enabled"]))
        if "elastic_percent" in data:
            self.elastic_spin.setValue(float(data["elastic_percent"]))

        # --- Cable mode (before update_2d so toggle state is correct) ---
        if "two_cable" in data:
            two = bool(data["two_cable"])
            self.cable2_check.setChecked(two)
            self.cable3_check.setChecked(not two)
            self._on_cable_toggle(two)
        if "num_cables" in data:
            self.cable_count_spin.setValue(int(data["num_cables"]))

        # --- Teardrop & cable3 cut ---
        if "teardrop_holes" in data:
            self.teardrop_check.setChecked(bool(data["teardrop_holes"]))
        if "cable3_cut_enabled" in data:
            self.cable3_cut_check.setChecked(bool(data["cable3_cut_enabled"]))
            self.cable3_cut_params_wrap.setVisible(
                (not self.params.two_cable) and bool(data["cable3_cut_enabled"])
            )
        if "cable3_cut_pos" in data:
            self.cable3_cut_pos_spin.setValue(float(data["cable3_cut_pos"]))
        if "cable3_cut_size" in data:
            self.cable3_cut_size_spin.setValue(float(data["cable3_cut_size"]))

        # --- Joint pose section (before update_2d so auto filler sizing sees it) ---
        if "pose_enabled" in data:
            # Block signals: _startup_build() rebuilds the scene anyway, and the
            # handler would run a full CAD build against half-restored params.
            self.pose_check.blockSignals(True)
            self.pose_check.setChecked(bool(data["pose_enabled"]))
            self.pose_check.blockSignals(False)
            self.pose_wrap.setVisible(bool(data["pose_enabled"]))
        if "filler_auto" in data:
            self.filler_auto_check.blockSignals(True)
            self.filler_auto_check.setChecked(bool(data["filler_auto"]))
            self.filler_auto_check.blockSignals(False)

        # --- update_2d to compute geometry and initialise ranges ---
        self.update_2d()

        # --- Fabrication params (re-applied after update_2d range init) ---
        if "extrusion" in data:
            v = float(data["extrusion"])
            v = max(self.extrusion_spin.minimum(), min(self.extrusion_spin.maximum(), v))
            self.extrusion_spin.setValue(v)
        if "cone_angle1" in data:
            v = float(data["cone_angle1"])
            v = max(self.cone1_spin.minimum(), min(self.cone1_spin.maximum(), v))
            self.cone1_spin.setValue(v)
        if "cone_angle2" in data:
            v = float(data["cone_angle2"])
            v = max(self.cone2_spin.minimum(), min(self.cone2_spin.maximum(), v))
            self.cone2_spin.setValue(v)
        if "show_cone_cuts" in data:
            self.show_cone_cuts_check.setChecked(bool(data["show_cone_cuts"]))
        if "tip_hole_pos" in data:
            self.tip_hole_pos_spin.setValue(float(data["tip_hole_pos"]))
        if "tip_hole_size" in data:
            self.tip_hole_size_spin.setValue(float(data["tip_hole_size"]))
        if "base_hole_pos" in data:
            self.base_hole_pos_spin.setValue(float(data["base_hole_pos"]))
        if "base_hole_size" in data:
            self.base_hole_size_spin.setValue(float(data["base_hole_size"]))

        # --- Simulation params ---
        if "sim_stiffness" in data:
            self.sim_stiffness_spin.setValue(float(data["sim_stiffness"]))
        if "sim_damping" in data:
            self.sim_damping_spin.setValue(float(data["sim_damping"]))
        if "sim_gravity" in data:
            self.sim_gravity_spin.setValue(float(data["sim_gravity"]))

        # --- Bend direction ---
        if "bend_dir_deg" in data:
            self.bend_dir_spin.setValue(float(data["bend_dir_deg"]))

        # Final update_2d with all params settled
        self.update_2d()

        # --- Joint angles (restored after update_2d rebuilds controls) ---
        joint_angles = data.get("joint_angles", [])
        for i, s in enumerate(getattr(self, "_joint_spins", [])):
            if i < len(joint_angles):
                s.setValue(float(joint_angles[i]))

        filler_widths = data.get("joint_filler_widths", [])
        for i, s in enumerate(getattr(self, "_joint_filler_widths", [])):
            if i < len(filler_widths):
                s.setValue(float(filler_widths[i]))
        filler_heights = data.get("joint_filler_heights", [])
        for i, s in enumerate(getattr(self, "_joint_filler_heights", [])):
            if i < len(filler_heights):
                s.setValue(float(filler_heights[i]))

        return True

    def closeEvent(self, event) -> None:
        self._save_params()
        super().closeEvent(event)

    def _apply_default_parameters(self, update_scene: bool = True) -> None:
        if hasattr(self, "_cone1_initialized"):
            delattr(self, "_cone1_initialized")
        if hasattr(self, "_cone2_initialized"):
            delattr(self, "_cone2_initialized")
        if hasattr(self, "_extrusion_initialized"):
            delattr(self, "_extrusion_initialized")
        defaults = Params()
        self.a_spin.setValue(defaults.a)
        self.b_spin.setValue(defaults.b)
        self.dtheta_spin.setValue(defaults.dtheta_deg)
        self.theta_max_spin.setValue(defaults.theta_max_pi)
        self.theta_max_slider.setValue(int(defaults.theta_max_pi * 10))
        self.p_spin.setValue(defaults.p)
        self.elastic_spin.setValue(defaults.elastic_percent)
        self.elastic_check.setChecked(defaults.elastic_enabled)
        self.cable2_check.setChecked(defaults.two_cable)
        self.cable3_check.setChecked(not defaults.two_cable)
        self.cable_count_spin.setValue(defaults.num_cables)
        self.teardrop_check.setChecked(defaults.teardrop_holes)
        self.cable3_cut_check.setChecked(defaults.cable3_cut_enabled)
        self.cable3_cut_pos_spin.setValue(defaults.cable3_cut_pos)
        self.cable3_cut_size_spin.setValue(defaults.cable3_cut_size)
        self.bend_dir_spin.setValue(defaults.bend_dir_deg)
        self.bend_dir_slider.setValue(int(defaults.bend_dir_deg * 10))
        self.cone1_spin.setValue(defaults.cone_angle1)
        self.cone1_slider.setValue(int(defaults.cone_angle1 * 10))
        self.cone2_spin.setValue(defaults.cone_angle2)
        self.cone2_slider.setValue(int(defaults.cone_angle2 * 10))
        self.show_cone_cuts_check.setChecked(False)
        self.tip_hole_pos_spin.setValue(defaults.tip_hole_pos)
        self.tip_hole_size_spin.setValue(defaults.tip_hole_size)
        self.base_hole_pos_spin.setValue(defaults.base_hole_pos)
        self.base_hole_size_spin.setValue(defaults.base_hole_size)
        self.sim_stiffness_spin.setValue(defaults.sim_stiffness)
        self.sim_stiffness_slider.setValue(int(defaults.sim_stiffness * 1000))
        self.sim_damping_spin.setValue(defaults.sim_damping)
        self.sim_damping_slider.setValue(int(defaults.sim_damping * 1000))
        self.sim_gravity_spin.setValue(defaults.gravity)
        self.sim_gravity_slider.setValue(int(defaults.gravity * 10))
        # Signals blocked: the pose handler would zero the joint spins and run a
        # CAD build, but the spins/widths/heights are set explicitly below.
        self.pose_check.blockSignals(True)
        self.pose_check.setChecked(defaults.pose_enabled)
        self.pose_check.blockSignals(False)
        self.pose_wrap.setVisible(defaults.pose_enabled)
        self.filler_auto_check.blockSignals(True)
        self.filler_auto_check.setChecked(defaults.filler_auto)
        self.filler_auto_check.blockSignals(False)
        self._on_cable_toggle(self.cable2_check.isChecked())
        self.update_2d()
        # re-apply cone1 defaults after update_2d range init
        self.cone1_spin.setValue(defaults.cone_angle1)
        self.cone1_slider.setValue(int(defaults.cone_angle1 * 10))
        self.update_2d()
        for s in self._joint_spins:
            s.setValue(0.0)
        default_widths, default_heights = self._joint_filler_defaults()
        for s, v in zip(getattr(self, "_joint_filler_widths", []), default_widths):
            s.setValue(v)
        for s, v in zip(getattr(self, "_joint_filler_heights", []), default_heights):
            s.setValue(v)
        if update_scene:
            self.update_scene()

    def reset_parameters(self) -> None:
        self._apply_default_parameters(update_scene=True)

    def update_2d(self) -> None:
        self.params.a = float(self.a_spin.value())
        self.params.b = float(self.b_spin.value())
        self.params.dtheta_deg = int(self.dtheta_spin.value())
        self.params.theta_max_pi = float(self.theta_max_spin.value())
        self.params.p = float(self.p_spin.value())
        self.params.elastic_percent = float(self.elastic_spin.value())
        self.params.elastic_enabled = self.elastic_check.isChecked()
        self.params.tip_hole_pos = float(self.tip_hole_pos_spin.value()) / 100.0
        self.params.tip_hole_size = float(self.tip_hole_size_spin.value())
        self.params.base_hole_pos = float(self.base_hole_pos_spin.value()) / 100.0
        self.params.base_hole_size = float(self.base_hole_size_spin.value())
        self.params.cable3_cut_enabled = self.cable3_cut_check.isChecked()
        self.params.cable3_cut_pos = float(self.cable3_cut_pos_spin.value())
        self.params.cable3_cut_size = float(self.cable3_cut_size_spin.value())
        self.params.num_cables = max(3, int(self.cable_count_spin.value()))
        self.params.teardrop_holes = self.teardrop_check.isChecked()
        self.params.sim_stiffness = float(self.sim_stiffness_spin.value())
        self.params.sim_damping = float(self.sim_damping_spin.value())
        self.params.gravity = float(self.sim_gravity_spin.value())

        turns = max(0.1, self.params.theta_max_pi / 2.0)
        theta_vals, r_vals, rc_vals, units_primary, units_mirror, unit_count = _build_polar_units(
            a=self.params.a,
            b=self.params.b,
            dtheta_deg=self.params.dtheta_deg,
            turns=turns,
            p=self.params.p,
        )
        primary, mirror = _build_unfold_polygons(
            a=self.params.a,
            b=self.params.b,
            dtheta_deg=self.params.dtheta_deg,
            p=self.params.p,
            turns=turns,
            unit_count=unit_count,
        )
        if not primary:
            return
        base_size = 2.0 * max(y for _x, y in primary[-1])
        thickness = max(0.1, base_size * 0.6)
        tip_size = 2.0 * max(y for _x, y in primary[0])
        theta_end = max(0.0, math.pi * self.params.theta_max_pi)
        eb = math.exp(2.0 * math.pi * self.params.b)
        taper_angle = 2.0 * math.atan(
            (self.params.b * (eb - 1.0))
            / (math.sqrt(self.params.b * self.params.b + 1.0) * (eb + 1.0))
        )
        self._taper_angle_deg = math.degrees(taper_angle)
        self._robot_length = max(x for x, _y in primary[-1])
        self._base_size = base_size
        self._tip_size = tip_size

        min_extrusion = max(0.0, self._base_size * 0.2)
        max_extrusion = max(min_extrusion, self._base_size)
        default_extrusion = max(0.0, self._base_size * 0.6)
        self.extrusion_spin.blockSignals(True)
        self.extrusion_slider.blockSignals(True)
        self.extrusion_spin.setRange(min_extrusion, max_extrusion)
        self.extrusion_slider.setRange(int(min_extrusion * 10), int(max_extrusion * 10))
        if hasattr(self, "_extrusion_initialized"):
            current_extrusion = float(self.extrusion_spin.value())
        else:
            current_extrusion = default_extrusion
        current_extrusion = max(min_extrusion, min(max_extrusion, current_extrusion))
        self.extrusion_spin.setValue(current_extrusion)
        self.extrusion_slider.setValue(int(current_extrusion * 10))
        self._extrusion_initialized = True
        self.extrusion_spin.blockSignals(False)
        self.extrusion_slider.blockSignals(False)
        self.params.extrusion = float(self.extrusion_spin.value())

        elastic_poly = None
        elastic_poly_mirror = None
        # Rays are based on taper angle and virtual tip
        eb = math.exp(2.0 * math.pi * self.params.b)
        c_factor = (1.0 - self.params.p) + self.params.p * eb
        l_vtip = (c_factor * self.params.a * math.sqrt(self.params.b**2 + 1.0)) / self.params.b
        elastic_angle = (self.params.elastic_percent / 100.0) * (taper_angle * 0.5)
        m = math.tan(elastic_angle) if elastic_angle != 0 else 0.0
        left_edge = (primary[0][0], primary[0][3])
        right_edge = (primary[-1][1], primary[-1][2])
        max_poly_x = max(x for poly in primary for x, _y in poly)
        ray_len = max(10.0, max_poly_x + l_vtip + 10.0)
        ray_start = (-l_vtip, 0.0)
        ray_upper_end = (-l_vtip + ray_len, m * ray_len)
        ray_lower_end = (-l_vtip + ray_len, -m * ray_len)
        self._ray_start = ray_start
        self._ray_upper_end = ray_upper_end
        self._ray_lower_end = ray_lower_end

        if self.params.elastic_enabled:
            upper_left = _line_segment_intersection(
                ray_start, ray_upper_end, left_edge[0], left_edge[1]
            )
            upper_right = _line_segment_intersection(
                ray_start, ray_upper_end, right_edge[0], right_edge[1]
            )
            q0_left = left_edge[1]
            q1_right = right_edge[1]
            if upper_left and upper_right:
                elastic_poly = [q0_left, upper_left, upper_right, q1_right]
                elastic_poly_mirror = [(x, -y) for x, y in elastic_poly]

        polys_all = primary + mirror
        if elastic_poly:
            polys_all.append(elastic_poly)
        if elastic_poly_mirror:
            polys_all.append(elastic_poly_mirror)

        self._polys_primary = primary
        self._polys_mirror = mirror
        self._elastic_poly = elastic_poly
        self._elastic_poly_mirror = elastic_poly_mirror
        self._polys_all = polys_all
        self._thickness = thickness

        self.taper_label.setText(f"Taper Angle: {self._taper_angle_deg:.2f}°")
        self.tip_label.setText(f"Tip Size: {tip_size:.2f} mm")
        self.base_label.setText(f"Base Size: {base_size:.2f} mm")
        self.length_label.setText(f"Robot Length: {self._robot_length:.2f} mm")
        self.units_label.setText(f"Num of Units: {len(primary)}")

        self.params.bend_dir_deg = float(self.bend_dir_spin.value())
        if max(0, len(primary) - 1) != len(self._joint_spins):
            self._rebuild_joint_controls()

        # Keep the per-joint filler sizes tracking the geometry while Auto is on.
        # update_2d runs before this panel exists during __init__, hence getattr.
        auto_check = getattr(self, "filler_auto_check", None)
        if auto_check is not None and auto_check.isChecked():
            self._sync_auto_filler_sizes()

        self._update_cone1_range()
        self._update_cone2_range()

        self._last_polar = (theta_vals, r_vals, rc_vals, units_primary, units_mirror, turns)
        self._last_cart = (primary, mirror, elastic_poly, elastic_poly_mirror)

        self._draw_polar(theta_vals, r_vals, rc_vals, units_primary, units_mirror, turns)
        self._draw_cartesian(primary, mirror, elastic_poly, elastic_poly_mirror)

    def _elastic_unit_polys(self):
        """Split the full-length elastic quad into per-unit quads (+ mirror).

        Returns (polys, mirrors), each a list of len(units) with None where the
        elastic layer does not cover the unit.
        """
        empty = [None] * len(self._polys_primary)
        if not self._elastic_poly:
            return empty, empty
        a, b, c, d = self._elastic_poly  # q0_left, upper_left, upper_right, q1_right

        def interp(p, q, x):
            if abs(q[0] - p[0]) < 1e-12:
                return p[1]
            t = (x - p[0]) / (q[0] - p[0])
            return p[1] + t * (q[1] - p[1])

        polys: List[List[Point2D] | None] = []
        mirrors: List[List[Point2D] | None] = []
        for k in range(len(self._polys_primary)):
            x0 = max(self._polys_primary[k][3][0], a[0])
            x1 = min(self._polys_primary[k][2][0], d[0])
            if x1 - x0 < 1e-9:
                polys.append(None)
                mirrors.append(None)
                continue
            quad = [
                (x0, interp(a, d, x0)),
                (x1, interp(a, d, x1)),
                (x1, interp(b, c, x1)),
                (x0, interp(b, c, x0)),
            ]
            polys.append(quad)
            mirrors.append([(x, -y) for x, y in quad])
        return polys, mirrors

    def _joint_filler_defaults(self):
        """Default (width, height) per joint for the 2-cable central-axis cylinder.

        Returns two lists in base->tip order (matching the joint control rows).

        Width is the *elastic core's* local diameter, not the rod's: every unit
        polygon runs from y=0 at both joint planes up to max(y) in the middle,
        so the revolved body pinches to r=0 at each joint and the only thing
        spanning the joint is the thin elastic core. Sizing the filler to the
        unit's outer diameter would make it reach a full radius into both
        neighbours and swallow them; the core diameter is exactly the gap that
        opens when the segments rotate about the pivot.

        Height is the rod's Z thickness, read live from the extrusion control.
        self._thickness is *not* that value -- update_2d sets it to
        base_size * 0.6 for the sketch drawing, while every CAD path extrudes by
        the extrusion control, so using it here made the fillers disagree with
        the body they sit in. The elastic layer also decides *whether* a joint
        gets a filler; where it is absent the fallback is the rod thickness.
        """
        n_joints = max(0, len(self._polys_primary) - 1)
        thickness = max(0.1, float(self.extrusion_spin.value()))
        elastic_polys, _elastic_mirror_polys = self._elastic_unit_polys()
        widths = []
        heights = []
        for j in range(n_joints):  # base->tip (UI order)
            tip_j = n_joints - 1 - j
            # The joint pivot sits on the axis (y=0), so the quad's |y| is the
            # core's radius there; the mirrored band gives the same magnitude.
            covered = tip_j < len(elastic_polys) and elastic_polys[tip_j]
            if covered:
                r_core = max(abs(y) for _x, y in elastic_polys[tip_j])
                widths.append(2.0 * r_core)
            else:
                widths.append(thickness)
            heights.append(thickness)
        return widths, heights

    def _build_joint_fillers(self, elastic_polys):
        """Knuckle fillers that keep the elastic rod continuous when bent.

        Bending rotates each elastic segment rigidly about the joint pivot, so
        adjacent segment end-faces open a wedge gap at every joint. A sphere
        (N-cable, pivot at rod center) or a cylinder along Z (2-cable, hinge
        pin) centered at the pivot fills that wedge. The 2-cable cylinder's
        diameter and Z-height come from the per-joint width/height controls.

        Returns a list aligned with units: entry j fills the joint between
        unit j and j+1 (built in the straight pose) and must be transformed
        with the same matrix as unit j. Entries are None where not needed.
        """
        try:
            import cadquery as cq
        except Exception:
            return [None] * len(self._polys_primary)
        n_units = len(self._polys_primary)
        fillers = [None] * n_units
        n_joints = n_units - 1
        widths = getattr(self, "_joint_filler_widths", [])
        heights = getattr(self, "_joint_filler_heights", [])
        for j in range(n_joints):
            eq = elastic_polys[j] if j < len(elastic_polys) else None
            nq = elastic_polys[j + 1] if j + 1 < len(elastic_polys) else None
            if not eq or not nq:
                continue
            x_e = eq[1][0]  # right edge x of unit j's elastic quad
            # only fill when the rod actually spans the joint
            if abs(nq[0][0] - x_e) > 1e-6:
                continue
            if self.params.two_cable:
                # Cylinder along Z (hinge axis). Controls are base->tip while
                # joint j here is tip->base, so map with n_joints - 1 - j.
                idx = n_joints - 1 - j
                width = widths[idx].value() if idx < len(widths) else 0.0
                height = heights[idx].value() if idx < len(heights) else 0.0
                if width <= 1e-9 or height <= 1e-9:
                    continue
                y_p = self._polys_primary[j][2][1]
                c = cq.Workplane("XY").circle(width / 2.0).extrude(height / 2.0, both=True)
                fillers[j] = c.translate((x_e, y_p, 0.0)).val()
            else:
                # Same basis as the 2-cable cylinder: the ball plugs the wedge
                # that opens in the elastic core, so its radius is the core's
                # radius at the joint -- the units pinch to r=0 there, which
                # makes the rod's outer radius far too big.
                r = max(abs(y) for _x, y in eq)
                if r < 1e-9:
                    continue
                s = cq.Workplane("XY").sphere(r)
                fillers[j] = s.translate((x_e, 0.0, 0.0)).val()
        return fillers

    @staticmethod
    def _halfspace_box(origin, normal, extent: float):
        """Big box occupying the half-space that `normal` points away to."""
        import cadquery as cq
        box = cq.Workplane("XY").box(extent, extent, extent, centered=(True, True, True))
        angle = math.degrees(math.atan2(normal[0], normal[2]))
        box = box.rotate((0, 0, 0), (0, 1, 0), angle)
        return box.translate((
            origin[0] + normal[0] * (extent / 2.0),
            origin[1] + normal[1] * (extent / 2.0),
            origin[2] + normal[2] * (extent / 2.0),
        ))

    def _joint_pose_angles(self) -> List[float]:
        return [float(s.value()) for s in getattr(self, "_joint_spins", [])]

    def _apply_joint_pose(self, main_units, elastic_units, extra_units=None):
        """Bend the straight per-unit Shape solids at the joints (serial chain).

        2-cable: hinge about the Z axis through the inner joint edge.
        N-cable: ball joint on the X axis, bend direction from bend_dir_spin.
        Each unit gets ONE composed gp_Trsf transform (not per-joint rotates).
        extra_units: optional unit-aligned list (e.g. joint fillers) transformed
        with the same per-unit matrices.
        """
        angles = self._joint_pose_angles()
        # CAD builds units from tip (small) to base (large).
        # Angles come in base-to-tip order → reverse for CAD processing.
        angles = list(reversed(angles))
        n_units = len(main_units)
        if n_units < 2 or len(angles) != n_units - 1:
            return main_units, elastic_units, extra_units
        if all(abs(a) < 1e-9 for a in angles):
            return main_units, elastic_units, extra_units

        import cadquery as cq
        from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

        if self.params.two_cable:
            pts = [
                [self._polys_primary[j][2][0], self._polys_primary[j][2][1], 0.0]
                for j in range(n_units - 1)
            ]
            dirs = [[0.0, 0.0, 1.0] for _ in range(n_units - 1)]
        else:
            beta = math.radians(float(self.bend_dir_spin.value()))
            pts = [
                [self._polys_primary[j][2][0], 0.0, 0.0]
                for j in range(n_units - 1)
            ]
            dirs = [[0.0, -math.sin(beta), math.cos(beta)] for _ in range(n_units - 1)]

        mats = [gp_Trsf() for _ in range(n_units)]  # identity
        for j, ang in enumerate(angles):
            if abs(ang) < 1e-9:
                continue
            pj, dj = pts[j], dirs[j]
            t = gp_Trsf()
            t.SetRotation(
                gp_Ax1(gp_Pnt(pj[0], pj[1], pj[2]), gp_Dir(dj[0], dj[1], dj[2])),
                math.radians(ang),
            )
            for k in range(j + 1, n_units):
                mats[k] = t.Multiplied(mats[k])
            # later joint frames move with the tail
            for jj in range(j + 1, n_units - 1):
                pts[jj] = list(_rotate_point_about_axis(tuple(pts[jj]), tuple(pj), tuple(dj), ang))
                dirs[jj] = list(_rodrigues(dirs[jj][0], dirs[jj][1], dirs[jj][2], dj[0], dj[1], dj[2], ang))

        def _tf(s, m):
            if s is None:
                return None
            return s.transformShape(cq.Matrix(m))

        main_out = [_tf(s, mats[k]) for k, s in enumerate(main_units)]
        elastic_out = [_tf(s, mats[k]) for k, s in enumerate(elastic_units)]
        extra_out = (
            [_tf(s, mats[k]) for k, s in enumerate(extra_units)]
            if extra_units is not None
            else None
        )
        return main_out, elastic_out, extra_out

    def _rebuild_joint_controls(self) -> None:
        prev = [float(s.value()) for s in getattr(self, "_joint_spins", [])]
        prev_widths = [float(s.value()) for s in getattr(self, "_joint_filler_widths", [])]
        prev_heights = [float(s.value()) for s in getattr(self, "_joint_filler_heights", [])]
        while self.joint_stack.count():
            page = self.joint_stack.widget(0)
            self.joint_stack.removeWidget(page)
            page.deleteLater()
        self._joint_spins = []
        self._joint_filler_widths = []
        self._joint_filler_heights = []
        n_joints = max(0, len(self._polys_primary) - 1)
        default_widths, default_heights = self._joint_filler_defaults()
        size_max = max(float(getattr(self, "_base_size", 1.0)) * 3.0, 1.0)
        auto = getattr(self, "filler_auto_check", None) is not None and self.filler_auto_check.isChecked()

        self.joint_selector.blockSignals(True)
        self.joint_selector.clear()
        for j in range(n_joints):
            self.joint_selector.addItem(f"J{j + 1}")
        self.joint_selector.blockSignals(False)

        for j in range(n_joints):
            # J1 = base joint (largest), J(N-1) = tip joint (smallest)
            # Controls and _joint_spins are stored base->tip, so index directly
            # (no reversal) and keep angle/width/height mappings consistent.
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)

            value = prev[j] if j < len(prev) else 0.0
            box, _slider, _row = self._add_double_control(
                grid, 0, "Angle", "(deg)", -90.0, 90.0, 0.5, 1, value, scale=10
            )
            self._joint_spins.append(box)

            # Auto mode always takes the freshly derived geometry; manual mode
            # preserves whatever the user last typed.
            w_val = default_widths[j] if auto else (
                prev_widths[j] if j < len(prev_widths) and prev_widths[j] > 0 else default_widths[j]
            )
            h_val = default_heights[j] if auto else (
                prev_heights[j] if j < len(prev_heights) and prev_heights[j] > 0 else default_heights[j]
            )
            # scale=100: the width/height boxes round-trip through their slider
            # (int(v * scale) in the valueChanged handler), and these two are the
            # only controls whose values come from the geometry. At scale=10 the
            # round-trip truncates an auto-derived 0.79 mm to 0.70 mm.
            wbox, _wsl, _row = self._add_double_control(
                grid, 1, "Width", "(mm)", 0.0, size_max, 0.1, 2, w_val, scale=100
            )
            self._joint_filler_widths.append(wbox)
            hbox, _hsl, _row = self._add_double_control(
                grid, 2, "Height", "(mm)", 0.0, size_max, 0.1, 2, h_val, scale=100
            )
            self._joint_filler_heights.append(hbox)

            self.joint_stack.addWidget(page)

        if n_joints > 0:
            self.joint_selector.setCurrentIndex(0)
            self.joint_stack.setCurrentIndex(0)

    def _reset_pose(self) -> None:
        for s in getattr(self, "_joint_spins", []):
            s.setValue(0.0)
        if hasattr(self, "bend_dir_spin"):
            self.bend_dir_spin.setValue(0.0)

    def _build_frustum_solid(self, angle_offset: float = 0.0):
        """Cable hole cutter along the p0->p1 centerline.

        Circular cross-section by default; teardrop cross-section (tip toward
        global +Z) when params.teardrop_holes is set. angle_offset is the
        rotation about +X that the caller applies afterwards — the teardrop
        profile is pre-rotated by -angle_offset so every hole's tip ends up
        pointing at global +Z.
        """
        if self._robot_length <= 1e-6:
            return None
        try:
            import cadquery as cq
        except Exception:
            return None
        tip_pos = (float(self.tip_hole_pos_spin.value()) / 100.0) if hasattr(self, "tip_hole_pos_spin") else self.params.tip_hole_pos
        tip_size = float(self.tip_hole_size_spin.value()) if hasattr(self, "tip_hole_size_spin") else self.params.tip_hole_size
        base_pos = (float(self.base_hole_pos_spin.value()) / 100.0) if hasattr(self, "base_hole_pos_spin") else self.params.base_hole_pos
        base_size = float(self.base_hole_size_spin.value()) if hasattr(self, "base_hole_size_spin") else self.params.base_hole_size
        y0 = max(0.0, min(1.0, tip_pos)) * (self._tip_size * 0.5)
        y1 = max(0.0, min(1.0, base_pos)) * (self._base_size * 0.5)
        tip_r = tip_size * 0.5
        base_r = base_size * 0.5
        # For small dtheta the tip unit reaches behind x = 0 (its angle at q0 is
        # over 90 deg): start the cutter there, on the same line and taper, so
        # the hole stays open at the tip.
        tip_min_x = min([0.0] + [q[0] for q in (self._polys_primary[0] if self._polys_primary else [])])
        ext = (-tip_min_x + max(tip_size, 0.1)) if tip_min_x < 0 else 0.0
        if ext > 0 and self._robot_length > 1e-9:
            y0 -= (y1 - y0) * ext / self._robot_length
            tip_r = max(1e-3, tip_r - (base_r - tip_r) * ext / self._robot_length)
        p0 = (-ext, y0, 0.0)
        p1 = (self._robot_length, y1, 0.0)
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        length_axis = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length_axis <= 1e-6:
            return None

        if self.params.teardrop_holes:
            # Teardrop profiles perpendicular to +X (YZ workplane), lofted tip -> base.
            # Circle radius r with a tangent apex at (0, sqrt(2)*r) pointing +Z.
            rot = math.radians(angle_offset)
            c_r, s_r = math.cos(rot), math.sin(rot)

            def td_pts(r: float):
                t = r / math.sqrt(2.0)
                apex = r * math.sqrt(2.0)
                pts = [(t, t), (0.0, -r), (-t, t), (0.0, apex)]
                return [(y * c_r + z * s_r, -y * s_r + z * c_r) for (y, z) in pts]

            q0 = td_pts(max(0.05, tip_r))
            q1 = td_pts(max(0.05, base_r))
            frustum = (
                cq.Workplane("YZ")
                .moveTo(*q0[0])
                .threePointArc(q0[1], q0[2])
                .lineTo(*q0[3])
                .close()
                .workplane(offset=length_axis)
                .moveTo(*q1[0])
                .threePointArc(q1[1], q1[2])
                .lineTo(*q1[3])
                .close()
                .loft(combine=True, ruled=True)
            )
        else:
            # Build frustum around +X axis in XY plane, then revolve around X
            profile = [(0.0, 0.0), (length_axis, 0.0), (length_axis, base_r), (0.0, tip_r)]
            frustum = (
                cq.Workplane("XY")
                .polyline(profile)
                .close()
                .revolve(360, (0, 0, 0), (1, 0, 0))
            )

        # Rotate frustum so its axis aligns with p0->p1, then translate so (length_axis,0,0) maps to p1
        vx, vy, vz = dx / length_axis, dy / length_axis, dz / length_axis
        ax, ay, az = 0.0, 0.0, 0.0
        dot = max(-1.0, min(1.0, vx))
        angle = math.degrees(math.acos(dot))
        if abs(angle) > 1e-6:
            # axis = cross([1,0,0], v) = (0, -vz, vy)
            ax, ay, az = 0.0, -vz, vy
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm > 1e-9:
                ax, ay, az = ax / norm, ay / norm, az / norm
                frustum = frustum.rotate((0, 0, 0), (ax, ay, az), angle)

        # compute rotated endpoint for (length_axis,0,0)
        def rot_vec(vx0, vy0, vz0):
            if abs(angle) <= 1e-6 or (ax == 0.0 and ay == 0.0 and az == 0.0):
                return (vx0, vy0, vz0)
            theta = math.radians(angle)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            kx, ky, kz = ax, ay, az
            dotp = kx * vx0 + ky * vy0 + kz * vz0
            rx = vx0 * cos_t + (ky * vz0 - kz * vy0) * sin_t + kx * dotp * (1.0 - cos_t)
            ry = vy0 * cos_t + (kz * vx0 - kx * vz0) * sin_t + ky * dotp * (1.0 - cos_t)
            rz = vz0 * cos_t + (kx * vy0 - ky * vx0) * sin_t + kz * dotp * (1.0 - cos_t)
            return (rx, ry, rz)

        end_rot = rot_vec(length_axis, 0.0, 0.0)
        tx = p1[0] - end_rot[0]
        ty = p1[1] - end_rot[1]
        tz = p1[2] - end_rot[2]
        frustum = frustum.translate((tx, ty, tz))
        return frustum

    def _build_cable_holes(self, angles: List[float]):
        """Union of cable-hole cutters at the given rotation angles about +X.

        Each instance is built with angle_offset=ang so teardrop tips stay
        pointing at global +Z after the rotation. Assembled as a compound
        (no boolean unions) for speed and robustness.
        """
        try:
            import cadquery as cq
        except Exception:
            return None
        shapes = []
        for ang in angles:
            f = self._build_frustum_solid(angle_offset=ang)
            if f is None:
                continue
            inst = f if abs(ang) < 1e-9 else f.rotate((0, 0, 0), (1, 0, 0), ang)
            shapes.append(inst.val())
        if not shapes:
            return None
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(shapes)])

    def _build_cone2_preview_solid(self):
        if not self.params.two_cable:
            return None
        try:
            import cadquery as cq
        except Exception:
            return None
        if self._robot_length <= 1e-6:
            return None
        cone1 = float(self.cone1_spin.value())
        cone2 = float(self.cone2_spin.value())
        if abs(cone2) <= 1e-6:
            return None
        thickness = max(0.1, float(self.extrusion_spin.value()))
        base_x = self._robot_length
        # point1: intersection of cone1 plane with +Z axis (x=0,y=0)
        alpha = -math.radians(cone1 * 0.5)
        point1 = (thickness * 0.5) + math.tan(alpha) * base_x
        p0 = (0.0, 0.0, point1)
        p1 = (self._robot_length, 0.0, thickness * 0.5)
        v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v2 = (0.0, self._base_size, 0.0)
        len_v1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
        len_v2 = max(1e-6, self._base_size)
        if len_v1 <= 1e-6:
            return None
        xdir = (v1[0] / len_v1, v1[1] / len_v1, v1[2] / len_v1)
        # normal = v1 x v2
        nx = v1[1] * v2[2] - v1[2] * v2[1]
        ny = v1[2] * v2[0] - v1[0] * v2[2]
        nz = v1[0] * v2[1] - v1[1] * v2[0]
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nlen <= 1e-9:
            return None
        normal = (nx / nlen, ny / nlen, nz / nlen)
        plane = cq.Plane(origin=p0, xDir=xdir, normal=normal)
        rect = (
            cq.Workplane(plane)
            .polyline([(0.0, 0.0), (len_v1, 0.0), (len_v1, len_v2), (0.0, len_v2)])
            .close()
            .extrude(self._base_size * 0.5)
        )
        # rotate around axis p0->p1 by cone2 (counterclockwise)
        rect = rect.rotate(p0, p1, -cone2)

        rect_xy = rect.mirror(mirrorPlane="XY")
        pair_xy = rect.union(rect_xy)
        rect_xz = pair_xy.mirror(mirrorPlane="XZ")
        pair_xz = pair_xy.union(rect_xz)
        return pair_xz

    def _build_cone_cut_regions(self):
        """Material that cone1/cone2 would actually remove from the uncut body.

        We build the body BEFORE any cutting (extrude every unfolded polygon),
        then intersect it with each cone cutter. The result is the exact chunk
        that gets shaved off — what we colour yellow in the preview so the user
        can see what ConeAngle1 / ConeAngle2 do.

        Returns (region1, region2) as cadquery Shape (or None each): region1 is
        the cone1 (base-face wedge) piece, region2 is the cone2 (base chamfer)
        piece. None for both when no cone cuts apply (3-cable or angles == 0).
        """
        if not self.params.two_cable:
            return None, None
        try:
            import cadquery as cq
        except Exception:
            return None, None
        if self._robot_length <= 1e-6 or not self._polys_primary:
            return None, None

        thickness = max(0.1, float(self.extrusion_spin.value()))

        # --- Uncut body: fuse every extruded unit (primary + mirror), no cutters.
        pieces = []
        for k in range(len(self._polys_primary)):
            wp = (
                cq.Workplane("XY")
                .polyline(self._polys_primary[k])
                .close()
                .extrude(thickness / 2.0, both=True)
            )
            if k < len(self._polys_mirror) and self._polys_mirror[k]:
                wm = (
                    cq.Workplane("XY")
                    .polyline(self._polys_mirror[k])
                    .close()
                    .extrude(thickness / 2.0, both=True)
                )
                wp = wp.union(wm)
            pieces.append(wp.val())
        if not pieces:
            return None, None
        body = pieces[0]
        for p in pieces[1:]:
            try:
                body = body.fuse(p)
            except Exception:
                pass

        # --- Cone1 wedge: two half-space clippers at the base face (same as CAD).
        region1 = None
        cone1 = float(self.cone1_spin.value())
        if self._robot_length > 1e-6 and cone1 > 1e-6:
            alpha = -math.radians(cone1 * 0.5)
            base_x = max(x for poly in self._polys_primary for x, _y in poly)
            half_z = thickness * 0.5
            extent = max(self._robot_length, self._base_size, thickness) * 10.0
            n1 = (math.sin(alpha), 0.0, math.cos(alpha))
            n2 = (math.sin(alpha), 0.0, -math.cos(alpha))
            for box in (
                self._halfspace_box((base_x, 0.0, half_z), n1, extent),
                self._halfspace_box((base_x, 0.0, -half_z), n2, extent),
            ):
                try:
                    r = body.intersect(box.val())
                except Exception:
                    r = None
                if r is not None and not r.isNull():
                    region1 = r if region1 is None else region1.fuse(r)

        # --- Cone2 chamfer: the rotated base solid intersected with the body.
        region2 = None
        cone2_solid = self._build_cone2_preview_solid()
        if cone2_solid is not None:
            cs = cone2_solid.val()
            if cs is not None and not cs.isNull():
                try:
                    r = body.intersect(cs)
                except Exception:
                    r = None
                if r is not None and not r.isNull():
                    region2 = r

        return region1, region2

    def _build_cable3_cut_solid(self):
        """Flat edge cuts between adjacent cable holes (weight reduction).

        Cross-section is shaved towards the inscribed regular N-gon:
        - Cut Position (%) = axial start of the cut (0 = full length, 50 = base half)
        - Cut Size (%)     = cut depth (0 = none, 20 = full inscribed N-gon)
        The apothem is clamped so flats never intersect the cable holes.
        """
        if self.params.two_cable or (not self.params.cable3_cut_enabled) or self._robot_length <= 1e-6:
            return None
        try:
            import cadquery as cq
        except Exception:
            return None

        n = max(3, int(self.params.num_cables))
        L = self._robot_length
        tip_r = max(1e-6, self._tip_size * 0.5)
        base_r = max(1e-6, self._base_size * 0.5)

        a1 = max(0.0, min(0.50, float(self.cable3_cut_pos_spin.value()) / 100.0))
        a2 = max(0.0, min(0.20, float(self.cable3_cut_size_spin.value()) / 100.0))
        if a2 <= 1e-9:
            return None
        x_start = 2.0 * a1 * L
        if x_start >= L - 1e-6:
            return None

        cos_step = math.cos(math.pi / n)
        depth_frac = a2 / 0.20  # 0..1 -> none .. inscribed N-gon

        def body_r(x: float) -> float:
            return tip_r + (base_r - tip_r) * (x / L)

        # Cable-hole centerline radius / hole radius along x (matches _build_frustum_solid)
        tip_pos = max(0.0, min(1.0, float(self.tip_hole_pos_spin.value()) / 100.0))
        base_pos = max(0.0, min(1.0, float(self.base_hole_pos_spin.value()) / 100.0))
        tip_hole_r = float(self.tip_hole_size_spin.value()) * 0.5
        base_hole_r = float(self.base_hole_size_spin.value()) * 0.5
        # Teardrop apex extends sqrt(2)*r from the hole center
        hole_extent = math.sqrt(2.0) if (hasattr(self, "teardrop_check") and self.teardrop_check.isChecked()) else 1.0
        rho0, rho1 = tip_pos * tip_r, base_pos * base_r

        hole_margin = 0.3  # mm, keep a wall around each cable hole

        def apothem(x: float) -> float:
            h_want = body_r(x) * (1.0 - depth_frac * (1.0 - cos_step))
            h_safe = (rho0 + (rho1 - rho0) * (x / L)) * cos_step \
                + (tip_hole_r + (base_hole_r - tip_hole_r) * (x / L)) * hole_extent + hole_margin
            return min(body_r(x), max(h_want, h_safe))

        h0 = apothem(x_start)
        h1 = apothem(L)
        slope = (h1 - h0) / (L - x_start)

        # Half-space beyond the tilted plane y = h0 + slope*(x - x_start), normal in XY plane
        nx, ny = -slope, 1.0
        norm = math.hypot(nx, ny)
        nx, ny = nx / norm, ny / norm
        extent = max(L, base_r, tip_r) * 10.0

        plane_box = cq.Workplane("XY").box(extent, extent, extent, centered=(True, True, True))
        gamma = math.degrees(math.atan2(-nx, ny))  # rotate +y onto the plane normal
        plane_box = plane_box.rotate((0, 0, 0), (0, 0, 1), gamma)
        plane_box = plane_box.translate((
            x_start + nx * (extent / 2.0),
            h0 + ny * (extent / 2.0),
            0.0,
        ))

        # Limit the cut to x in [x_start, L + margin]
        x_len = (L * 1.1) - x_start
        slab = cq.Workplane("XY").box(x_len, extent, extent, centered=(True, True, True))
        slab = slab.translate((x_start + x_len / 2.0, 0.0, 0.0))

        cutter0 = plane_box.intersect(slab)

        # With 5+ cables neighbouring flats overlap, and OCCT's cut with a
        # compound of overlapping tools leaves fragments: fuse them first.
        tool = None
        for k in range(n):
            phi = (k + 0.5) * 360.0 / n  # flats sit between adjacent cable holes
            inst = cutter0.rotate((0, 0, 0), (1, 0, 0), phi).val()
            tool = inst if tool is None else tool.fuse(inst)
        return cq.Workplane("XY").newObject([tool])

    def _export_stl_single_threaded(self, shape, path: str, tolerance: float) -> bool:
        """Export a CadQuery Shape/Workplane to binary STL, meshing single-threaded.

        OCCT's default STL path meshes in parallel (`BRepMesh_IncrementalMesh`
        with parallel=True). Once VTK's OpenGL context is loaded into the same
        process (the 3D preview), one of those background meshing threads hits
        an access violation — a native library interaction unrelated to the
        geometry (the same shape meshes fine without VTK). Meshing on the
        calling thread (parallel=False) avoids it entirely.
        """
        import cadquery as cq
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer
        s = shape
        if isinstance(s, cq.Workplane):
            s = s.val()
        wrapped = s.wrapped
        # relative=False: `tolerance` is an absolute deflection in mm (the value
        # comes from _stl_mesh_tolerance). Note: it does NOT change the resulting
        # file size much — the body is mostly flat per-unit quad faces that always
        # triangulate to ~2 triangles regardless of deflection, and the few curved
        # faces (elastic frustum, cone) hit the 0.1 rad angular cap first.
        BRepMesh_IncrementalMesh(wrapped, tolerance, False, 0.1, False)
        writer = StlAPI_Writer()
        return bool(writer.Write(wrapped, path))

    def _stl_mesh_tolerance(self) -> float:
        """STL tessellation deflection (mm) sized to the model.

        Scaled to the body so surface accuracy is roughly constant across models
        (a 300 mm body gets ~0.09 mm, a small one gets finer). The value is an
        absolute mm deflection, NOT a ratio — see the call site in
        _export_stl_single_threaded. This controls accuracy, not mesh size; the
        size is dominated by the flat per-unit faces, so a few tens of MB binary
        STL is expected and later halved by vtkDecimatePro in the preview.
        """
        length = max(1.0, float(getattr(self, "_robot_length", 1.0)))
        return max(0.05, length * 0.0003)

    def update_scene(self) -> None:
        # Heavy 3D rebuild on demand
        self._ensure_vtk_ready()
        if not self._vtk_initialized:
            return
        parts = self._build_cad_parts()
        if parts is None:
            return
        main_solid, elastic_solid = parts
        if main_solid is None:
            return
        try:
            import cadquery as cq
        except Exception:
            self.vtk_placeholder.setText("3D Preview unavailable (CadQuery not ready)")
            self.vtk_placeholder.setVisible(True)
            return
        solid_shapes = [main_solid.val()]
        if elastic_solid is not None:
            solid_shapes.append(elastic_solid.val())
        solid = cq.Compound.makeCompound(solid_shapes)

        out_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(out_dir, exist_ok=True)
        preview_path = os.path.join(out_dir, "preview_live.stl")
        try:
            if not self._export_stl_single_threaded(solid, preview_path, self._stl_mesh_tolerance()):
                raise RuntimeError("OCCT STL write returned failure")
        except Exception as exc:
            self.vtk_placeholder.setText(f"3D Preview export failed: {exc}")
            self.vtk_placeholder.setVisible(True)
            return

        reader = vtkSTLReader()
        reader.SetFileName(preview_path)
        reader.Update()
        poly = reader.GetOutput()
        # Thin out the mesh before rendering: a ~7.7 MB STL (hundreds of
        # thousands of triangles) plus edge rendering over-stresses an Intel
        # iGPU and triggers a Windows display-driver TDR (black screen + crash).
        # Halve the triangle count (visual difference is negligible).
        decimate = vtkDecimatePro()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(0.5)
        decimate.PreserveTopologyOn()
        decimate.Update()
        normals = vtkPolyDataNormals()
        normals.SetInputData(decimate.GetOutput())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        triangulate = vtkTriangleFilter()
        triangulate.SetInputData(normals.GetOutput())
        triangulate.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(triangulate.GetOutput())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.55, 0.73, 0.90)
        actor.GetProperty().SetOpacity(0.95)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.4, 0.55, 0.7)
        actor.GetProperty().BackfaceCullingOff()
        actor.GetProperty().SetAmbient(0.35)
        actor.GetProperty().SetDiffuse(0.65)
        actor.GetProperty().SetSpecular(0.1)
        actor.GetProperty().SetInterpolationToFlat()
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)

        # Debug: show x1/y1 and x2/y2 intersection points as small spheres
        posed = any(abs(a) > 1e-9 for a in self._joint_pose_angles())
        site_points = None
        if self._polys_primary and not posed:
            last_poly = self._polys_primary[-1]
            if len(last_poly) >= 4:
                p0_line = (0.0, self.params.tip_hole_pos * self._tip_size * 0.5)
                p1_line = (self._robot_length, self.params.base_hole_pos * self._base_size * 0.5)
                dx = p1_line[0] - p0_line[0]
                dy = p1_line[1] - p0_line[1]
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    dx, dy = 1.0, 0.0
                length = (dx * dx + dy * dy) ** 0.5
                dx /= length
                dy /= length
                L = 1e6
                line_a = (p0_line[0] - dx * L, p0_line[1] - dy * L)
                line_b = (p0_line[0] + dx * L, p0_line[1] + dy * L)
                left_edge = (last_poly[3], last_poly[0])
                right_edge = (last_poly[2], last_poly[1])
                left_hit = _line_segment_intersection(line_a, line_b, left_edge[0], left_edge[1])
                right_hit = _line_segment_intersection(line_a, line_b, right_edge[0], right_edge[1])
                if left_hit and right_hit:
                    x1, y1 = left_hit
                    x2, y2 = right_hit
                    site_points = (x1, y1, x2, y2)
                else:
                    x1 = left_edge[0][0]
                    x2 = right_edge[0][0]
                    if abs(p1_line[0] - p0_line[0]) < 1e-9:
                        y1 = p0_line[1]
                        y2 = p1_line[1]
                    else:
                        slope = (p1_line[1] - p0_line[1]) / (p1_line[0] - p0_line[0])
                        y1 = p0_line[1] + slope * (x1 - p0_line[0])
                        y2 = p0_line[1] + slope * (x2 - p0_line[0])
                    site_points = (x1, y1, x2, y2)

        if site_points is not None:
            x1, y1, x2, y2 = site_points
            for (sx, sy, color) in ((x1, y1, (1.0, 0.2, 0.2)), (x2, y2, (0.2, 0.8, 0.2))):
                sphere = vtkSphereSource()
                sphere.SetCenter(sx, sy, 0.0)
                sphere.SetRadius(max(0.2, 0.01 * max(self._base_size, 1.0)))
                sphere.SetThetaResolution(24)
                sphere.SetPhiResolution(24)
                sphere.Update()
                s_mapper = vtkPolyDataMapper()
                s_mapper.SetInputData(sphere.GetOutput())
                s_actor = vtkActor()
                s_actor.SetMapper(s_mapper)
                s_actor.GetProperty().SetColor(*color)
                s_actor.GetProperty().SetOpacity(0.9)
                self.renderer.AddActor(s_actor)

        frustum_poly = None
        frustum_path = os.path.join(out_dir, "frustum_live.stl")
        frustum_solid = None if posed else self._build_frustum_solid()
        if frustum_solid is not None and not self.params.teardrop_holes:
            try:
                import cadquery as cq
                cq.exporters.export(frustum_solid.val(), frustum_path)
                f_reader = vtkSTLReader()
                f_reader.SetFileName(frustum_path)
                f_reader.Update()
                frustum_poly = f_reader.GetOutput()
            except Exception:
                frustum_poly = None

            f_normals = vtkPolyDataNormals()
            f_normals.SetInputData(frustum_poly)
            f_normals.ConsistencyOn()
            f_normals.AutoOrientNormalsOn()
            f_normals.SplittingOff()
            f_normals.Update()
            f_triangulate = vtkTriangleFilter()
            f_triangulate.SetInputData(f_normals.GetOutput())
            f_triangulate.Update()
            f_mapper = vtkPolyDataMapper()
            f_mapper.SetInputData(f_triangulate.GetOutput())
            angles = self._cable_angles()
            for ang in angles:
                f_actor = vtkActor()
                f_actor.SetMapper(f_mapper)
                f_actor.GetProperty().SetColor(1.0, 0.55, 0.2)
                f_actor.GetProperty().SetOpacity(0.5)
                f_actor.GetProperty().EdgeVisibilityOn()
                f_actor.GetProperty().SetEdgeColor(0.85, 0.4, 0.15)
                f_actor.GetProperty().BackfaceCullingOff()
                if abs(ang) > 1e-6:
                    t = vtkTransform()
                    t.RotateWXYZ(ang, 1.0, 0.0, 0.0)
                    f_actor.SetUserTransform(t)
                self.renderer.AddActor(f_actor)
        elif self.params.teardrop_holes and not posed:
            # Teardrop tips must stay at global +Z: build one solid per angle
            # (pre-rotated by -ang) instead of reusing a single mapper.
            for idx, ang in enumerate(self._cable_angles()):
                f_solid = self._build_frustum_solid(angle_offset=ang)
                if f_solid is None:
                    continue
                f_path = os.path.join(out_dir, f"frustum_live_{idx}.stl")
                try:
                    import cadquery as cq
                    cq.exporters.export(f_solid.val(), f_path)
                    f_reader = vtkSTLReader()
                    f_reader.SetFileName(f_path)
                    f_reader.Update()
                    f_poly = f_reader.GetOutput()
                except Exception:
                    continue
                f_normals = vtkPolyDataNormals()
                f_normals.SetInputData(f_poly)
                f_normals.ConsistencyOn()
                f_normals.AutoOrientNormalsOn()
                f_normals.SplittingOff()
                f_normals.Update()
                f_triangulate = vtkTriangleFilter()
                f_triangulate.SetInputData(f_normals.GetOutput())
                f_triangulate.Update()
                f_mapper = vtkPolyDataMapper()
                f_mapper.SetInputData(f_triangulate.GetOutput())
                f_actor = vtkActor()
                f_actor.SetMapper(f_mapper)
                f_actor.GetProperty().SetColor(1.0, 0.55, 0.2)
                f_actor.GetProperty().SetOpacity(0.5)
                f_actor.GetProperty().EdgeVisibilityOn()
                f_actor.GetProperty().SetEdgeColor(0.85, 0.4, 0.15)
                f_actor.GetProperty().BackfaceCullingOff()
                if abs(ang) > 1e-6:
                    t = vtkTransform()
                    t.RotateWXYZ(ang, 1.0, 0.0, 0.0)
                    f_actor.SetUserTransform(t)
                self.renderer.AddActor(f_actor)

        cone2_poly = None
        cone2_path = os.path.join(out_dir, "cone2_preview.stl")
        cone2_solid = self._build_cone2_preview_solid()
        if cone2_solid is not None:
            try:
                if not self._export_stl_single_threaded(cone2_solid, cone2_path, self._stl_mesh_tolerance()):
                    raise RuntimeError("OCCT STL write returned failure")
                c_reader = vtkSTLReader()
                c_reader.SetFileName(cone2_path)
                c_reader.Update()
                cone2_poly = c_reader.GetOutput()
            except Exception:
                cone2_poly = None

        if self._show_cone2_preview and cone2_poly is not None:
            c_normals = vtkPolyDataNormals()
            c_normals.SetInputData(cone2_poly)
            c_normals.ConsistencyOn()
            c_normals.AutoOrientNormalsOn()
            c_normals.SplittingOff()
            c_normals.Update()
            c_triangulate = vtkTriangleFilter()
            c_triangulate.SetInputData(c_normals.GetOutput())
            c_triangulate.Update()
            c_mapper = vtkPolyDataMapper()
            c_mapper.SetInputData(c_triangulate.GetOutput())
            c_actor = vtkActor()
            c_actor.SetMapper(c_mapper)
            c_actor.GetProperty().SetColor(0.9, 0.8, 0.2)
            c_actor.GetProperty().SetOpacity(0.35)
            c_actor.GetProperty().EdgeVisibilityOn()
            c_actor.GetProperty().SetEdgeColor(0.8, 0.7, 0.2)
            c_actor.GetProperty().BackfaceCullingOff()
            self.renderer.AddActor(c_actor)

        # Yellow highlight of the exact material cone1/cone2 would remove.
        # Reuse the already-exported live STL shape as the "uncut" reference so we
        # don't need a separate boolean pass on the geometry; if no cone cuts, skip.
        if (
            getattr(self, "show_cone_cuts_check", None) is not None
            and self.show_cone_cuts_check.isChecked()
        ):
            try:
                region1, region2 = self._build_cone_cut_regions()
            except Exception:
                region1 = region2 = None
            for region, tint in ((region1, (0.95, 0.55, 0.05)), (region2, (0.55, 0.8, 0.05))):
                if region is None:
                    continue
                # region is a cadquery Shape; wrap + write to a temp STL.
                try:
                    import cadquery as cq
                    r_path = os.path.join(out_dir, "cone_cut_region.stl")
                    cq.exporters.export(cq.Workplane("XY").newObject([region]), r_path)
                except Exception:
                    continue
                try:
                    r_reader = vtkSTLReader()
                    r_reader.SetFileName(r_path)
                    r_reader.Update()
                    r_poly = r_reader.GetOutput()
                    r_normals = vtkPolyDataNormals()
                    r_normals.SetInputData(r_poly)
                    r_normals.ConsistencyOn()
                    r_normals.AutoOrientNormalsOn()
                    r_normals.SplittingOff()
                    r_normals.Update()
                    r_tri = vtkTriangleFilter()
                    r_tri.SetInputData(r_normals.GetOutput())
                    r_tri.Update()
                    r_map = vtkPolyDataMapper()
                    r_map.SetInputData(r_tri.GetOutput())
                    r_act = vtkActor()
                    r_act.SetMapper(r_map)
                    r_act.GetProperty().SetColor(*tint)
                    r_act.GetProperty().SetOpacity(0.6)
                    r_act.GetProperty().EdgeVisibilityOn()
                    r_act.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
                    r_act.GetProperty().BackfaceCullingOff()
                    self.renderer.AddActor(r_act)
                except Exception:
                    pass

        # Place the coordinate axes at the base (bottom) joint's end-face center
        # (x=_robot_length) instead of the model tip, and orient them to match the
        # cross-section XY plane used at XML export so tendon positions read the same
        # here as in the exported model. The export frame is
        # translate(-robot_length,0,0) then rotate(+90 deg about Y), which maps:
        #   XML X = CAD Z (thickness), XML Y = CAD Y (width), XML Z = -CAD X (chain).
        # Rotating the triad -90 deg about Y aligns its Z with the chain (base->tip)
        # and its X/Y with the cross-section, matching that export XY plane.
        self._axes_actor.SetPosition(self._robot_length, 0.0, 0.0)
        self._axes_actor.SetOrientation(0.0, -90.0, 0.0)
        self.renderer.AddActor(self._axes_actor)
        if hasattr(self, "_axes_legend"):
            self.renderer.AddActor(self._axes_legend)
        if not self._camera_initialized:
            self.renderer.ResetCamera()
            self._camera_initialized = True
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
        self._has_3d = True
        self.vtk_placeholder.setVisible(False)

    def _cone1_planes(
        self, cone1_deg: float, thickness: float, base_x: float
    ) -> List[vtkPlane]:
        planes: List[vtkPlane] = []
        if base_x <= 1e-6 or cone1_deg <= 1e-6:
            return planes
        alpha = -math.radians(cone1_deg * 0.5)
        half_z = thickness * 0.5

        p1 = vtkPlane()
        p1.SetOrigin(base_x, 0.0, half_z)
        p1.SetNormal(math.sin(alpha), 0.0, math.cos(alpha))
        planes.append(p1)

        p2 = vtkPlane()
        p2.SetOrigin(base_x, 0.0, -half_z)
        p2.SetNormal(math.sin(alpha), 0.0, -math.cos(alpha))
        planes.append(p2)
        return planes

    def _update_cone2_range(self) -> None:
        max_angle_deg = max(0.0, 4.0 * self._taper_angle_deg)
        current = self.cone2_spin.value()
        self.cone2_spin.setRange(0.0, max_angle_deg)
        self.cone2_slider.setRange(0, int(max_angle_deg * 10))
        if current > max_angle_deg:
            self.cone2_spin.setValue(max_angle_deg)

    def _update_cone1_range(self) -> None:
        max_angle_deg = 0.0
        extrusion_val = float(self.extrusion_spin.value()) if hasattr(self, "extrusion_spin") else self.params.extrusion
        if self._robot_length > 1e-6:
            max_angle_deg = math.degrees(
                2.0 * math.atan((extrusion_val * 0.5) / self._robot_length)
            )
        max_angle_deg = max(0.0, max_angle_deg)
        current = self.cone1_spin.value()
        self.cone1_spin.setRange(0.0, max_angle_deg)
        self.cone1_slider.setRange(0, int(max_angle_deg * 10))
        if current > max_angle_deg:
            self.cone1_spin.setValue(max_angle_deg)

    def _clip_closed_by_plane(
        self, poly: vtkPolyData, plane: vtkPlane, inside_out: int
    ) -> vtkPolyData:
        tri = vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        clean = vtkCleanPolyData()
        clean.SetInputData(tri.GetOutput())
        clean.Update()

        planes = vtkPlaneCollection()
        planes.AddItem(plane)
        closed = vtkClipClosedSurface()
        closed.SetInputData(clean.GetOutput())
        closed.SetClippingPlanes(planes)
        closed.GenerateFacesOn()
        closed.SetInsideOut(inside_out)
        closed.Update()

        clean2 = vtkCleanPolyData()
        clean2.SetInputData(closed.GetOutput())
        clean2.Update()
        return clean2.GetOutput()


    def reset_view(self) -> None:
        if not self._vtk_initialized:
            return
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _draw_polar(
        self,
        theta_vals: List[float],
        r_vals: List[float],
        rc_vals: List[float],
        units_primary: List[Tuple[List[float], List[float]]],
        units_mirror: List[Tuple[List[float], List[float]]],
        turns: float,
    ) -> None:
        self._draw_polar_on(
            self.polar_ax,
            theta_vals,
            r_vals,
            rc_vals,
            units_primary,
            units_mirror,
            turns,
        )
        self.polar_canvas.draw_idle()

    def _draw_polar_on(
        self,
        ax,
        theta_vals: List[float],
        r_vals: List[float],
        rc_vals: List[float],
        units_primary: List[Tuple[List[float], List[float]]],
        units_mirror: List[Tuple[List[float], List[float]]],
        turns: float,
    ) -> None:
        ax.clear()
        ax.set_title("SpiRobs Design")

        ax.plot(theta_vals, r_vals, color="#1f77b4", linewidth=2.0)
        rc_end = max(0.0, 2.0 * math.pi * turns - 2.0 * math.pi)
        rc_theta = [t for t in theta_vals if t <= rc_end + 1e-12]
        rc_r = rc_vals[: len(rc_theta)]
        ax.plot(rc_theta, rc_r, color="#ff7f0e", linewidth=2.0)
        for theta_poly, r_poly in units_primary:
            ax.fill(
                theta_poly,
                r_poly,
                color="#9ecae1",
                alpha=0.35,
                edgecolor="#6baed6",
                linewidth=0.6,
            )
        for theta_poly, r_poly in units_mirror:
            ax.fill(
                theta_poly,
                r_poly,
                color="#a1d99b",
                alpha=0.35,
                edgecolor="#74c476",
                linewidth=0.6,
            )
        ax.grid(True, alpha=0.3)
        # Limit radial ticks for faster rendering
        if ax.get_rmax() > 0:
            ax.set_rticks([ax.get_rmax() * 0.5, ax.get_rmax()])

    def _draw_cartesian(
        self,
        primary: List[List[Point2D]],
        mirror: List[List[Point2D]],
        elastic_poly: List[Point2D] | None,
        elastic_poly_mirror: List[Point2D] | None,
    ) -> None:
        self._draw_cartesian_on(
            self.cart_ax,
            primary,
            mirror,
            elastic_poly,
            elastic_poly_mirror,
            self.params.elastic_enabled,
        )
        self.cart_canvas.draw_idle()

    def _draw_cartesian_on(
        self,
        ax,
        primary: List[List[Point2D]],
        mirror: List[List[Point2D]],
        elastic_poly: List[Point2D] | None,
        elastic_poly_mirror: List[Point2D] | None,
        draw_rays: bool,
    ) -> None:
        ax.clear()
        ax.set_title("Unfolded (Cartesian)")
        for poly in primary:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            ax.fill(xs, ys, color="#9ecae1", alpha=0.35, edgecolor="#6baed6", linewidth=0.6)
        for poly in mirror:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            ax.fill(xs, ys, color="#a1d99b", alpha=0.35, edgecolor="#74c476", linewidth=0.6)
        if elastic_poly:
            xs = [p[0] for p in elastic_poly]
            ys = [p[1] for p in elastic_poly]
            ax.fill(xs, ys, color="#ff7f0e", alpha=0.28, edgecolor="#ff7f0e", linewidth=0.9)
        if elastic_poly_mirror:
            xs = [p[0] for p in elastic_poly_mirror]
            ys = [p[1] for p in elastic_poly_mirror]
            ax.fill(xs, ys, color="#ff7f0e", alpha=0.28, edgecolor="#ff7f0e", linewidth=0.9)
        if draw_rays and self._ray_start and self._ray_upper_end and self._ray_lower_end:
            def _clip_ray(start: Point2D, end: Point2D, x_max: float) -> Point2D:
                sx, sy = start
                ex, ey = end
                if ex <= x_max:
                    return end
                if abs(ex - sx) < 1e-9:
                    return (x_max, sy)
                t = (x_max - sx) / (ex - sx)
                return (sx + t * (ex - sx), sy + t * (ey - sy))

            end_u = _clip_ray(self._ray_start, self._ray_upper_end, self._robot_length)
            end_l = _clip_ray(self._ray_start, self._ray_lower_end, self._robot_length)
            ax.plot(
                [self._ray_start[0], end_u[0]],
                [self._ray_start[1], end_u[1]],
                color="#9aa0a6",
                linewidth=0.6,
            )
            ax.plot(
                [self._ray_start[0], end_l[0]],
                [self._ray_start[1], end_l[1]],
                color="#9aa0a6",
                linewidth=0.6,
            )
        ax.set_aspect("equal", adjustable="box")
        if primary:
            min_x = min(min(x for x, _y in poly) for poly in primary)
            max_x = max(max(x for x, _y in poly) for poly in primary)
            if self._ray_start and self._ray_upper_end and self._ray_lower_end:
                min_x = min(min_x, self._ray_start[0])
                max_x = max(max_x, self._ray_upper_end[0], self._ray_lower_end[0])
            pad_x = 0.05 * (max_x - min_x + 1e-6)
            ax.set_xlim(min_x - pad_x, max_x + pad_x)
            y_limit = max(self._base_size * 0.75, 1e-6)
            ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.grid(True, alpha=0.2)

    def save_image(self) -> None:
        out_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        polar_path = os.path.join(out_dir, f"polar_{ts}.png")
        cart_path = os.path.join(out_dir, f"cartesian_{ts}.png")
        preview_path = os.path.join(out_dir, f"preview_3d_{ts}.png")
        window_pdf = os.path.join(out_dir, f"window_{ts}.pdf")
        self.polar_fig.savefig(polar_path, dpi=200)
        self.cart_fig.savefig(cart_path, dpi=200)
        # Prefer VTK capture to avoid black OpenGL grabs
        try:
            self.vtk_widget.GetRenderWindow().Render()
            w2i = vtkWindowToImageFilter()
            w2i.SetInput(self.vtk_widget.GetRenderWindow())
            w2i.ReadFrontBufferOn()
            w2i.Update()
            writer = vtkPNGWriter()
            writer.SetFileName(preview_path)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
        except Exception:
            grab = self.vtk_widget.grab()
            grab.save(preview_path, "PNG")

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(window_pdf)
        printer.setPageMargins(QMarginsF(8, 8, 8, 8))
        painter = QPainter(printer)
        try:
            target = painter.viewport()
            source = self.frameGeometry()
            source.moveTo(0, 0)
            scaled = source.size()
            scaled.scale(target.size(), Qt.KeepAspectRatio)
            painter.setViewport(target.x(), target.y(), scaled.width(), scaled.height())
            painter.setWindow(source)
            self.render(painter, QPoint(0, 0))
        finally:
            painter.end()

        # Vector 2D + raster 3D report
        report_path = os.path.join(out_dir, f"report_{ts}.pdf")
        if self._last_polar and self._last_cart:
            fig = Figure(figsize=(11, 6))
            canvas = FigureCanvas(fig)
            gs = fig.add_gridspec(
                3,
                3,
                width_ratios=[1.2, 1.2, 1.0],
                height_ratios=[0.18, 1.0, 1.0],
            )
            ax_header = fig.add_subplot(gs[0, :])
            ax_polar = fig.add_subplot(gs[1, 0], projection="polar")
            ax_cart = fig.add_subplot(gs[2, 0:2])
            ax_3d = fig.add_subplot(gs[1:, 2])

            theta_vals, r_vals, rc_vals, units_primary, units_mirror, turns = self._last_polar
            self._draw_polar_on(
                ax_polar,
                theta_vals,
                r_vals,
                rc_vals,
                units_primary,
                units_mirror,
                turns,
            )
            primary, mirror, elastic_poly, elastic_poly_mirror = self._last_cart
            self._draw_cartesian_on(
                ax_cart,
                primary,
                mirror,
                elastic_poly,
                elastic_poly_mirror,
                self.params.elastic_enabled,
            )
            ax_header.axis("off")
            ax_header.text(
                0.01,
                0.6,
                "OpenSpiRob Design",
                fontsize=14,
                fontweight="bold",
                color="#333333",
                va="center",
            )
            ax_header.text(
                0.01,
                0.15,
                f"a={self.params.a:.4f} mm  b={self.params.b:.4f}  Δθ={self.params.dtheta_deg}°  θmax={self.params.theta_max_pi:.1f}π  p={self.params.p:.2f}",
                fontsize=9,
                color="#555555",
                va="center",
            )
            ax_header.text(
                0.01,
                -0.25,
                f"Taper={self._taper_angle_deg:.2f}°  Tip={2.0 * max(p[1] for p in self._polys_primary[0]):.2f} mm  Base={self._base_size:.2f} mm  Length={self._robot_length:.2f} mm  Units={len(self._polys_primary)}",
                fontsize=9,
                color="#555555",
                va="center",
            )

            ax_3d.set_title("3D Model")
            ax_3d.axis("off")
            img = mpimg.imread(preview_path)
            ax_3d.imshow(img)
            fig.tight_layout()
            fig.savefig(report_path, dpi=300)

    def export_xml(self) -> None:
        try:
            import cadquery as cq
        except Exception:
            return
        self._save_params()
        self.params.sim_stiffness = float(self.sim_stiffness_spin.value())
        self.params.sim_damping = float(self.sim_damping_spin.value())
        self.params.gravity = float(self.sim_gravity_spin.value())
        out_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(out_dir, exist_ok=True)

        # Build only the rightmost unit without elastic layer
        if not self._polys_primary:
            return
        num_units = max(1, len(self._polys_primary))
        cable_mode = (2 if self.params.two_cable else max(3, int(self.params.num_cables)))

        # Ask the user for per-tendon settings before exporting.
        dlg = TendonConfigDialog(
            num_tendons=cable_mode,
            num_joints=max(0, num_units - 1),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        tendon_configs = dlg.get_configs()

        # Only create the per-export folder after the user confirms with OK, so
        # closing the dialog without exporting leaves no empty xml_<ts> folder
        # behind (and the timestamp reflects when the export actually started).
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xml_dir = os.path.join(out_dir, f"xml_{ts}")
        os.makedirs(xml_dir, exist_ok=True)

        if self.params.two_cable:
            thickness = max(0.1, float(self.extrusion_spin.value()))
            solid = None
            right_primary = self._polys_primary[-1]
            right_mirror = self._polys_mirror[-1] if self._polys_mirror else None
            for poly in (right_primary, right_mirror):
                if not poly:
                    continue
                wp = cq.Workplane("XY").polyline(poly).close().extrude(thickness / 2.0, both=True)
                solid = wp if solid is None else solid.union(wp)
            if solid is None:
                return

            # Apply cone1 cut on the unit
            cone1 = float(self.cone1_spin.value())
            if self._robot_length > 1e-6 and cone1 > 1e-6:
                alpha = -math.radians(cone1 * 0.5)
                try:
                    base_x = solid.val().BoundingBox().xmax
                except Exception:
                    base_x = self._robot_length
                half_z = thickness * 0.5
                extent = max(self._robot_length, self._base_size, thickness) * 10.0

                n1 = (math.sin(alpha), 0.0, math.cos(alpha))
                n2 = (math.sin(alpha), 0.0, -math.cos(alpha))

                def _cut_halfspace(workpiece, origin, normal):
                    box = cq.Workplane("XY").box(extent, extent, extent, centered=(True, True, True))
                    angle = math.degrees(math.atan2(normal[0], normal[2]))
                    box = box.rotate((0, 0, 0), (0, 1, 0), angle)
                    box = box.translate((
                        origin[0] + normal[0] * (extent / 2.0),
                        origin[1] + normal[1] * (extent / 2.0),
                        origin[2] + normal[2] * (extent / 2.0),
                    ))
                    return workpiece.cut(box)

                solid = _cut_halfspace(solid, (base_x, 0.0, half_z), n1)
                solid = _cut_halfspace(solid, (base_x, 0.0, -half_z), n2)

            # Apply cone2 cut on the unit
            cone2_solid = self._build_cone2_preview_solid()
            if cone2_solid is not None:
                solid = solid.cut(cone2_solid)

        else:
            solid = None
            right_primary = self._polys_primary[-1]
            wp = (
                cq.Workplane("XY")
                .polyline(right_primary)
                .close()
                .revolve(360, (0, 0, 0), (1, 0, 0))
            )
            solid = wp if solid is None else solid.union(wp)
            if solid is None:
                return

            # Apply frustum holes for 3-cable unit
            holes = self._build_cable_holes(self._cable_angles())
            if holes is not None:
                solid = solid.cut(holes)

            # Apply additional 3-cable manufacturing cuts
            cable3_cut = self._build_cable3_cut_solid()
            if cable3_cut is not None:
                solid = solid.cut(cable3_cut)

        # Transform: translate along x by -robot_length, then rotate about y by 90 deg
        solid = solid.translate((-self._robot_length, 0.0, 0.0))
        solid = solid.rotate((0, 0, 0), (0, 1, 0), 90)

        stl_name = "baselink.stl"
        stl_path = os.path.join(xml_dir, stl_name)
        cq.exporters.export(solid.val(), stl_path)

        xml_path = os.path.join(xml_dir, "robot.xml")
        # compute unit height from rightmost quad (x-axis segment)
        unit_height = 0.0
        if self._polys_primary:
            pts = [p for p in self._polys_primary[-1] if abs(p[1]) < 1e-6]
            if len(pts) >= 2:
                unit_height = abs(pts[1][0] - pts[0][0])
            else:
                unit_height = abs(self._polys_primary[-1][2][0] - self._polys_primary[-1][3][0])
        unit_height = max(1e-6, unit_height)
        # compute cable sites based on last unit
        site_points = None
        if self._polys_primary:
            last_poly = self._polys_primary[-1]
            if len(last_poly) >= 4:
                p0_line = (0.0, self.params.tip_hole_pos * self._tip_size * 0.5)
                p1_line = (self._robot_length, self.params.base_hole_pos * self._base_size * 0.5)
                dx = p1_line[0] - p0_line[0]
                dy = p1_line[1] - p0_line[1]
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    dx, dy = 1.0, 0.0
                length = (dx * dx + dy * dy) ** 0.5
                dx /= length
                dy /= length
                L = 1e6
                line_a = (p0_line[0] - dx * L, p0_line[1] - dy * L)
                line_b = (p0_line[0] + dx * L, p0_line[1] + dy * L)
                left_edge = (last_poly[3], last_poly[0])
                right_edge = (last_poly[2], last_poly[1])
                left_hit = _line_segment_intersection(line_a, line_b, left_edge[0], left_edge[1])
                right_hit = _line_segment_intersection(line_a, line_b, right_edge[0], right_edge[1])
                if left_hit and right_hit:
                    x1, y1 = left_hit
                    x2, y2 = right_hit
                    site_points = (x1, y1, x2, y2)
                else:
                    # Fallback: compute y on the line at left/right x positions
                    x1 = left_edge[0][0]
                    x2 = right_edge[0][0]
                    if abs(p1_line[0] - p0_line[0]) < 1e-9:
                        # vertical in x, use end y values
                        y1 = p0_line[1]
                        y2 = p1_line[1]
                    else:
                        slope = (p1_line[1] - p0_line[1]) / (p1_line[0] - p0_line[0])
                        y1 = p0_line[1] + slope * (x1 - p0_line[0])
                        y2 = p0_line[1] + slope * (x2 - p0_line[0])
                    site_points = (x1, y1, x2, y2)
        gamma = math.exp(self.params.b * math.radians(self.params.dtheta_deg))
        joint_type = "hinge" if self.params.two_cable else "ball"
        joint_angles = self._joint_pose_angles()
        bend_dir = float(self.bend_dir_spin.value())
        from xml_generator import generate_mujoco_xml
        try:
            generate_mujoco_xml(
                xml_path,
                stl_name=stl_name,
                unit_height=unit_height,
                scale=gamma,
                num_units=num_units,
                joint_type=joint_type,
                joint_limit_deg=self.params.dtheta_deg,
                robot_length=self._robot_length,
                site_points=site_points,
                cable_mode=cable_mode,
                joint_stiffness=self.params.sim_stiffness,
                joint_damping=self.params.sim_damping,
                gravity=self.params.gravity,
                joint_angles=joint_angles,
                bend_dir_deg=bend_dir,
                extrusion_depth=(thickness if self.params.two_cable else 0.0),
                tendon_configs=tendon_configs,
            )
        except Exception as exc:
            print(f"[Export XML] failed: {exc}")
            return

    def export_cad(self) -> None:
        try:
            import cadquery as cq
        except Exception:
            return
        self._save_params()
        parts = self._build_cad_parts()
        if parts is None:
            return
        main_solid, elastic_solid = parts
        if main_solid is None:
            return
        out_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_path = os.path.join(out_dir, f"spi_rob_{ts}.step")
        stl_path = os.path.join(out_dir, f"spi_rob_{ts}.stl")
        solids = [main_solid.val()]
        if elastic_solid is not None:
            solids.append(elastic_solid.val())
        step_compound = cq.Compound.makeCompound(solids)
        cq.exporters.export(step_compound, step_path)
        # STL: mesh single-threaded (parallel=False). With VTK's OpenGL context
        # loaded, OCCT's default parallel meshing trips an access violation; a
        # size-scaled tolerance also keeps the file a sensible size.
        self._export_stl_single_threaded(step_compound, stl_path, self._stl_mesh_tolerance())

    def _build_cad_parts(self):
        try:
            import cadquery as cq
        except Exception:
            return None

        n_units = len(self._polys_primary)
        if n_units == 0:
            return None
        elastic_polys, elastic_mirror_polys = self._elastic_unit_polys()

        def _shape_of(obj):
            return None if obj is None else obj.val()

        def _wrap(shapes):
            shapes = [s for s in shapes if s is not None and not s.isNull()]
            if not shapes:
                return None
            return cq.Workplane("XY").newObject([cq.Compound.makeCompound(shapes)])

        def _bbox_overlap(a, b, tol=1e-6):
            ba, bb = a.BoundingBox(), b.BoundingBox()
            return (
                ba.xmin <= bb.xmax + tol and ba.xmax >= bb.xmin - tol
                and ba.ymin <= bb.ymax + tol and ba.ymax >= bb.ymin - tol
                and ba.zmin <= bb.zmax + tol and ba.zmax >= bb.zmin - tol
            )

        def _iter_cutters(cutters):
            for c in cutters:
                cs = _shape_of(c)
                if cs is not None and not cs.isNull():
                    yield cs

        def _cut_per_unit(shapes, cutters):
            # Posed path: cut each unit, skipping cutters whose bbox misses it.
            tools = list(_iter_cutters(cutters))
            if not tools:
                return shapes
            out = []
            for s in shapes:
                if s is None:
                    out.append(None)
                    continue
                for cs in tools:
                    if not _bbox_overlap(s, cs):
                        continue
                    try:
                        s2 = s.cut(cs)
                        if s2 is not None and not s2.isNull():
                            s = s2
                    except Exception:
                        pass
                out.append(s)
            return out

        def _cut_global(shapes, cutters):
            # Straight path: assemble once, then one boolean per cutter.
            shapes = [s for s in shapes if s is not None and not s.isNull()]
            if not shapes:
                return None
            comp = cq.Compound.makeCompound(shapes)
            for cs in _iter_cutters(cutters):
                try:
                    comp2 = comp.cut(cs)
                    if comp2 is not None and not comp2.isNull():
                        comp = comp2
                except Exception:
                    pass
            return comp

        posed = any(abs(a) > 1e-9 for a in self._joint_pose_angles())

        if self.params.two_cable:
            thickness = max(0.1, float(self.extrusion_spin.value()))

            main_units = []
            for k in range(n_units):
                wp = cq.Workplane("XY").polyline(self._polys_primary[k]).close().extrude(thickness / 2.0, both=True)
                if k < len(self._polys_mirror) and self._polys_mirror[k]:
                    wm = cq.Workplane("XY").polyline(self._polys_mirror[k]).close().extrude(thickness / 2.0, both=True)
                    wp = wp.union(wm)
                main_units.append(wp.val())

            elastic_units = []
            for k in range(n_units):
                eu = None
                if elastic_polys[k]:
                    eu = cq.Workplane("XY").polyline(elastic_polys[k]).close().extrude(thickness / 2.0, both=True)
                    if elastic_mirror_polys[k]:
                        eum = cq.Workplane("XY").polyline(elastic_mirror_polys[k]).close().extrude(thickness / 2.0, both=True)
                        eu = eu.union(eum)
                elastic_units.append(None if eu is None else eu.val())

            # Cone1 clipping half-spaces (same planes as before)
            cone_boxes = []
            cone1 = float(self.cone1_spin.value())
            if self._robot_length > 1e-6 and cone1 > 1e-6:
                alpha = -math.radians(cone1 * 0.5)
                base_x = max(x for poly in self._polys_primary for x, _y in poly)
                half_z = thickness * 0.5
                extent = max(self._robot_length, self._base_size, thickness) * 10.0
                n1 = (math.sin(alpha), 0.0, math.cos(alpha))
                n2 = (math.sin(alpha), 0.0, -math.cos(alpha))
                cone_boxes = [
                    self._halfspace_box((base_x, 0.0, half_z), n1, extent),
                    self._halfspace_box((base_x, 0.0, -half_z), n2, extent),
                ]

            frustum = self._build_cable_holes([0.0, 180.0])
            cone2_solid = self._build_cone2_preview_solid()
            cutters = cone_boxes + [frustum, cone2_solid]

            if posed:
                main_units = _cut_per_unit(main_units, cutters)
                elastic_units = _cut_per_unit(elastic_units, cutters)
                fillers = self._build_joint_fillers(elastic_polys)
                # Same cutters as the units: an auto-sized filler spans the local
                # diameter, so it would otherwise poke through the cone faces.
                fillers = _cut_per_unit(fillers, cutters)
                main_units, elastic_units, fillers = self._apply_joint_pose(
                    main_units, elastic_units, fillers
                )
                main = _wrap(main_units)
                elastic = _wrap(elastic_units + (fillers or []))
            else:
                main = _wrap([_cut_global(main_units, cutters)])
                elastic = _wrap([_cut_global(elastic_units, cutters)])
            if main is None:
                return None
            return (main, elastic)

        main_units = []
        for k in range(n_units):
            wp = (
                cq.Workplane("XY")
                .polyline(self._polys_primary[k])
                .close()
                .revolve(360, (0, 0, 0), (1, 0, 0))
            )
            main_units.append(wp.val())

        elastic_units = []
        for k in range(n_units):
            eu = None
            if elastic_polys[k]:
                eu = (
                    cq.Workplane("XY")
                    .polyline(elastic_polys[k])
                    .close()
                    .revolve(360, (0, 0, 0), (1, 0, 0))
                )
            elastic_units.append(None if eu is None else eu.val())

        frustum = self._build_cable_holes(self._cable_angles())
        cable3_cut = self._build_cable3_cut_solid()
        cutters = [frustum, cable3_cut]

        if posed:
            main_units = _cut_per_unit(main_units, cutters)
            elastic_units = _cut_per_unit(elastic_units, cutters)
            fillers = self._build_joint_fillers(elastic_polys)
            fillers = _cut_per_unit(fillers, cutters)
            main_units, elastic_units, fillers = self._apply_joint_pose(
                main_units, elastic_units, fillers
            )
            main = _wrap(main_units)
            elastic = _wrap(elastic_units + (fillers or []))
        else:
            main = _wrap([_cut_global(main_units, cutters)])
            elastic = _wrap([_cut_global(elastic_units, cutters)])
        if main is None:
            return None
        return (main, elastic)


def _hide_gl_symbol_warnings() -> None:
    """Silence cosmetic OpenGL-loader noise ("Symbol '<glX>' is not defined").

    When the app runs on a virtual display (e.g. Sunlogin's Oray Indirect
    Display Driver) or any surface that can't provide a hardware desktop-OpenGL
    context, Windows falls back to the software "GDI Generic" OpenGL 1.1
    renderer. That ICD lacks every modern GL entry point, so the loader prints
    one "Symbol '<glX>' is not defined" line per missing function (six of them
    appear right after startup). They are harmless — the preview just renders in
    software — but they clutter the terminal.

    We redirect the process stdout/stderr through a daemon reader that forwards
    every line unchanged EXCEPT ones matching this specific "symbol ... is not
    defined" pattern. Real output is preserved; in particular Python
    NameError tracebacks ("name '...' is not defined") do NOT contain "symbol"
    so they are never dropped. Non-Windows platforms are untouched.
    """
    if os.name != "nt" or getattr(sys, "_gl_warn_filter_on", False):
        return
    sys._gl_warn_filter_on = True
    import threading

    def _is_gl_loader_line(line: bytes) -> bool:
        lower = line.lower()
        return b"symbol" in lower and b"not defined" in lower

    def _filter_stream(src_fd: int, keep_fd: int) -> None:
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, src_fd)
        os.close(write_fd)

        def _reader() -> None:
            buf = b""
            while True:
                try:
                    chunk = os.read(read_fd, 4096)
                except OSError:
                    return
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if _is_gl_loader_line(line):
                        continue
                    os.write(keep_fd, line + b"\n")
            if buf and not _is_gl_loader_line(buf):
                os.write(keep_fd, buf)

        threading.Thread(target=_reader, daemon=True).start()

    sys.stdout.flush()
    sys.stderr.flush()
    # Duplicate the real fds as the forwarding target BEFORE hijacking them.
    _filter_stream(1, os.dup(1))
    _filter_stream(2, os.dup(2))


def main() -> None:
    # Loading the desktop-OpenGL ICD over a virtual display prints benign
    # "Symbol ... is not defined" lines; drop just those so the console is clean.
    _hide_gl_symbol_warnings()

    # Native crashes (segfault / OpenGL driver crash) cannot be caught by
    # try/except, so dump the C+Python call stack to exports/crash.log instead.
    import faulthandler
    crash_log = os.path.join(os.path.dirname(__file__), "exports", "crash.log")
    try:
        os.makedirs(os.path.dirname(crash_log), exist_ok=True)
        faulthandler.enable(open(crash_log, "a", encoding="utf-8"))
    except Exception:
        faulthandler.enable()

    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    # Avoid forcing a fixed 3.2 Core profile: on Intel iGPUs a hard-coded core
    # OpenGL context is a known trigger for the VTK preview to black-screen and
    # crash after repeated Generate calls. Let Qt/VTK negotiate a compatible
    # context instead.
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()









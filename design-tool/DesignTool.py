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
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolygon, vtkPolyData, vtkPlane, vtkPlaneCollection
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkTriangleFilter, vtkClipPolyData, vtkPolyDataNormals
from vtkmodules.vtkFiltersSources import vtkPlaneSource, vtkArrowSource, vtkCubeSource, vtkSphereSource
from vtkmodules.vtkFiltersModeling import vtkLinearExtrusionFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter, vtkClipClosedSurface
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
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
    sim_stiffness: float = 0.5
    sim_damping: float = 0.2
    two_cable: bool = True


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
        self._camera_initialized = False
        self._extrusion_initialized = False
        self._axes_actor = vtkAxesActor()
        self._axes_actor.SetTotalLength(20.0, 20.0, 20.0)
        self._axes_actor.SetShaftTypeToCylinder()
        self._axes_actor.SetCylinderRadius(0.02)
        self._axes_actor.SetAxisLabels(False)
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
            form_2d, row, "a", "(mm)", 0.1, 20.0, 0.01, 4, self.params.a, scale=1000
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

        cable_row = QHBoxLayout()
        cable_row.setContentsMargins(0, 0, 0, 0)
        cable_row.setSpacing(8)
        self.cable_toggle = ToggleSwitch(self.params.two_cable, on_color="#2f6fb8", off_color="#2fb86f")
        self.cable_label = QLabel("2-cable")
        cable_row.addWidget(self.cable_toggle)
        cable_row.addWidget(self.cable_label)
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
        form_3d.addWidget(self.cable2_wrap, row, 0, 1, 3)
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
            1.0,
            0.01,
            3,
            self.params.sim_stiffness,
            scale=1000,
        )
        self.sim_damping_spin, self.sim_damping_slider, sim_row = self._add_double_control(
            form_sim,
            sim_row,
            "Damping",
            "",
            0.0,
            1.0,
            0.01,
            3,
            self.params.sim_damping,
            scale=1000,
        )
        panel_layout.addLayout(form_sim)

        label_actions = QLabel("Actions")
        label_actions.setStyleSheet("font-weight:600; color:#666;")

        panel_layout.addStretch(1)

        panel_layout.addWidget(label_actions)
        self.update_btn = QPushButton("Generate 3D Model")
        panel_layout.addWidget(self.update_btn)
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
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        self.export_cad_btn.clicked.connect(self.export_cad)
        self.export_xml_btn.clicked.connect(self.export_xml)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.cable_toggle.toggled.connect(self._on_cable_toggle)
        self._on_cable_toggle(self.cable_toggle.isChecked())
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
            self.tip_hole_pos_slider,
            self.tip_hole_size_slider,
            self.base_hole_pos_slider,
            self.base_hole_size_slider,
        ):
            w.valueChanged.connect(self.schedule_update)
        self.elastic_check.toggled.connect(lambda _v: self.schedule_update())

        self.update_2d()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_splitter_sizes)
        self._ensure_vtk_ready()
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
                interactor.Start()
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
        self._update_timer.start(120)
        self._update_cone1_range()
        self._update_cone2_range()

    def _on_cable_toggle(self, checked: bool) -> None:
        self.params.two_cable = bool(checked)
        self.cable_label.setText("2-cable" if checked else "3-cable")
        self.cable2_wrap.setVisible(checked)
        self.update_scene()

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
        self.vtk_widget.setUpdatesEnabled(False)
        self.polar_canvas.setUpdatesEnabled(False)
        self.cart_canvas.setUpdatesEnabled(False)
        self._resize_timer.start(120)

    def _finish_resize(self) -> None:
        self.vtk_widget.setUpdatesEnabled(True)
        self.polar_canvas.setUpdatesEnabled(True)
        self.cart_canvas.setUpdatesEnabled(True)
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

        
    def reset_parameters(self) -> None:
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
        self.cable_toggle.setChecked(defaults.two_cable)
        self.cone1_spin.setValue(defaults.cone_angle1)
        self.cone1_slider.setValue(int(defaults.cone_angle1 * 10))
        self.cone2_spin.setValue(defaults.cone_angle2)
        self.cone2_slider.setValue(int(defaults.cone_angle2 * 10))
        self.tip_hole_pos_spin.setValue(defaults.tip_hole_pos)
        self.tip_hole_size_spin.setValue(defaults.tip_hole_size)
        self.base_hole_pos_spin.setValue(defaults.base_hole_pos)
        self.base_hole_size_spin.setValue(defaults.base_hole_size)
        self.sim_stiffness_spin.setValue(defaults.sim_stiffness)
        self.sim_stiffness_slider.setValue(int(defaults.sim_stiffness * 1000))
        self.sim_damping_spin.setValue(defaults.sim_damping)
        self.sim_damping_slider.setValue(int(defaults.sim_damping * 1000))
        self._on_cable_toggle(self.cable_toggle.isChecked())
        self.update_2d()
        # re-apply cone1 defaults after update_2d range init
        self.cone1_spin.setValue(defaults.cone_angle1)
        self.cone1_slider.setValue(int(defaults.cone_angle1 * 10))
        self.update_2d()
        self.update_scene()

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
        current_extrusion = default_extrusion
        if current_extrusion < min_extrusion or current_extrusion > max_extrusion:
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
        if not hasattr(self, "_extrusion_initialized"):
            self.params.extrusion = thickness
            self.extrusion_spin.setValue(thickness)
            self.extrusion_slider.setValue(int(thickness * 10))
            self._extrusion_initialized = True

        self.taper_label.setText(f"Taper Angle: {self._taper_angle_deg:.2f}°")
        self.tip_label.setText(f"Tip Size: {tip_size:.2f} mm")
        self.base_label.setText(f"Base Size: {base_size:.2f} mm")
        self.length_label.setText(f"Robot Length: {self._robot_length:.2f} mm")
        self.units_label.setText(f"Num of Units: {len(primary)}")

        self._update_cone1_range()
        self._update_cone2_range()

        self._last_polar = (theta_vals, r_vals, rc_vals, units_primary, units_mirror, turns)
        self._last_cart = (primary, mirror, elastic_poly, elastic_poly_mirror)

        self._draw_polar(theta_vals, r_vals, rc_vals, units_primary, units_mirror, turns)
        self._draw_cartesian(primary, mirror, elastic_poly, elastic_poly_mirror)

    def _build_frustum_solid(self):
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
        p0 = (0.0, y0, 0.0)
        p1 = (self._robot_length, y1, 0.0)
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        length_axis = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length_axis <= 1e-6:
            return None

        # Build frustum around +X axis in XY plane, then revolve around X
        profile = [(0.0, 0.0), (length_axis, 0.0), (length_axis, base_size * 0.5), (0.0, tip_size * 0.5)]
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

    def update_scene(self) -> None:
        # Heavy 3D rebuild on demand
        self._ensure_vtk_ready()
        if not self._vtk_initialized:
            return
        solid = self._build_cad_solid()
        if solid is None:
            self.vtk_placeholder.setText("3D Preview unavailable (CadQuery not ready)")
            self.vtk_placeholder.setVisible(True)
            return

        out_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(out_dir, exist_ok=True)
        preview_path = os.path.join(out_dir, "preview_live.stl")
        try:
            import cadquery as cq
            cq.exporters.export(solid.val(), preview_path)
        except Exception as exc:
            self.vtk_placeholder.setText(f"3D Preview export failed: {exc}")
            self.vtk_placeholder.setVisible(True)
            return

        reader = vtkSTLReader()
        reader.SetFileName(preview_path)
        reader.Update()
        poly = reader.GetOutput()
        normals = vtkPolyDataNormals()
        normals.SetInputData(poly)
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
        actor.GetProperty().SetInterpolationToPhong()
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)

        # Debug: show x1/y1 and x2/y2 intersection points as small spheres
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
        frustum_solid = self._build_frustum_solid()
        if frustum_solid is not None:
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
            if self.params.two_cable:
                angles = [0.0, 180.0]
            else:
                angles = [0.0, 120.0, 240.0]
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

        cone2_poly = None
        cone2_path = os.path.join(out_dir, "cone2_preview.stl")
        cone2_solid = self._build_cone2_preview_solid()
        if cone2_solid is not None:
            try:
                import cadquery as cq
                cq.exporters.export(cone2_solid.val(), cone2_path)
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

        self.renderer.AddActor(self._axes_actor)
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
        out_dir = os.path.join(os.getcwd(), "exports")
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
        out_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xml_dir = os.path.join(out_dir, f"xml_{ts}")
        os.makedirs(xml_dir, exist_ok=True)

        # Build only the rightmost unit without elastic layer
        if not self._polys_primary:
            return
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
        num_units = max(1, len(self._polys_primary))
        joint_type = "hinge" if self.params.two_cable else "ball"
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
                cable_mode=3 if not self.params.two_cable else 2,
            )
        except Exception as exc:
            print(f"[Export XML] failed: {exc}")
            return

    def export_cad(self) -> None:
        try:
            import cadquery as cq
        except Exception:
            return
        solid = self._build_cad_solid()
        if solid is None:
            return
        out_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_path = os.path.join(out_dir, f"spi_rob_{ts}.step")
        stl_path = os.path.join(out_dir, f"spi_rob_{ts}.stl")
        cq.exporters.export(solid.val(), step_path)
        cq.exporters.export(solid.val(), stl_path)

    def _build_cad_solid(self):
        try:
            import cadquery as cq
        except Exception:
            return None

        if self.params.two_cable:
            thickness = max(0.1, float(self.extrusion_spin.value()))

            solid = None
            for poly in (self._polys_primary + self._polys_mirror):
                wp = cq.Workplane("XY").polyline(poly).close().extrude(thickness / 2.0, both=True)
                solid = wp if solid is None else solid.union(wp)
            if self._elastic_poly:
                wp = cq.Workplane("XY").polyline(self._elastic_poly).close().extrude(thickness / 2.0, both=True)
                solid = wp if solid is None else solid.union(wp)
            if self._elastic_poly_mirror:
                wp = cq.Workplane("XY").polyline(self._elastic_poly_mirror).close().extrude(thickness / 2.0, both=True)
                solid = wp if solid is None else solid.union(wp)
            if solid is None:
                return None
            # Apply cone1 clipping to CAD export
            cone1 = float(self.cone1_spin.value())
            if self._robot_length > 1e-6 and cone1 > 1e-6:
                alpha = -math.radians(cone1 * 0.5)
                # Use actual model bounds for base_x to avoid offset errors
                try:
                    base_x = solid.val().BoundingBox().xmax
                except Exception:
                    base_x = self._robot_length
                half_z = thickness * 0.5
                extent = max(self._robot_length, self._base_size, thickness) * 10.0

                # Same two planes as 3D preview
                n1 = (math.sin(alpha), 0.0, math.cos(alpha))
                n2 = (math.sin(alpha), 0.0, -math.cos(alpha))

                def _cut_halfspace(workpiece, origin, normal):
                    # Build a large box and remove the side pointed by normal
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

            frustum = self._build_frustum_solid()
            if frustum is not None:
                angles = [0.0, 180.0]
                holes = None
                for ang in angles:
                    inst = frustum if ang == 0.0 else frustum.rotate((0, 0, 0), (1, 0, 0), ang)
                    holes = inst if holes is None else holes.union(inst)
                if holes is not None:
                    solid = solid.cut(holes)

            cone2_solid = self._build_cone2_preview_solid()
            if cone2_solid is not None:
                solid = solid.cut(cone2_solid)

            return solid

        solid = None
        for poly in self._polys_primary:
            wp = (
                cq.Workplane("XY")
                .polyline(poly)
                .close()
                .revolve(360, (0, 0, 0), (1, 0, 0))
            )
            solid = wp if solid is None else solid.union(wp)
        if self._elastic_poly:
            wp = (
                cq.Workplane("XY")
                .polyline(self._elastic_poly)
                .close()
                .revolve(360, (0, 0, 0), (1, 0, 0))
            )
            solid = wp if solid is None else solid.union(wp)
        frustum = self._build_frustum_solid()
        if frustum is not None:
            angles = [0.0, 120.0, 240.0]
            holes = None
            for ang in angles:
                inst = frustum if ang == 0.0 else frustum.rotate((0, 0, 0), (1, 0, 0), ang)
                holes = inst if holes is None else holes.union(inst)
            if holes is not None:
                solid = solid.cut(holes)
        return solid


def main() -> None:
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
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

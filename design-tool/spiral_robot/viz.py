import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

from .spiral import SpiralSamples


def plot_spiral(
    samples: SpiralSamples,
    title: str = "Logarithmic Spiral",
    show_points: bool = True,
    show_line: bool = True,
    equal_aspect: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot spiral samples in Cartesian coordinates."""
    fig, ax = plt.subplots(figsize=(7, 7))

    if show_line:
        ax.plot(samples.x, samples.y, linewidth=1.5, label="Spiral")
    if show_points:
        ax.scatter(samples.x, samples.y, s=10, alpha=0.7, label="Samples")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def _polar_to_cart(theta: float, r: float) -> Tuple[float, float]:
    return (r * math.cos(theta), r * math.sin(theta))


def _cart_to_polar(x: float, y: float) -> Tuple[float, float]:
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2.0 * math.pi
    return theta, r


def _reflect_point_across_line(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[float, float]:
    # Reflect point p across line through a-b (Cartesian).
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        return p
    t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)
    projx = ax + t * vx
    projy = ay + t * vy
    rx = 2.0 * projx - px
    ry = 2.0 * projy - py
    return rx, ry


def _build_forward_units(
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
    theta = 0.0
    dtheta = math.radians(max(1, int(dtheta_deg)))

    theta_vals: List[float] = []
    r_vals: List[float] = []
    rc_vals: List[float] = []
    units_primary: List[Tuple[List[float], List[float]]] = []
    units_mirror: List[Tuple[List[float], List[float]]] = []

    eb = math.exp(2.0 * math.pi * b)
    c_factor = p + (1.0 - p) * eb

    while theta <= theta_end + 1e-12:
        r = a * math.exp(b * theta)
        rc = c_factor * r

        theta_vals.append(theta)
        r_vals.append(r)
        rc_vals.append(rc)
        theta += dtheta

    for i in range(len(theta_vals) - 1):
        t0 = theta_vals[i]
        t1 = theta_vals[i + 1]
        if t1 > rc_end + 1e-12:
            break
        r0 = r_vals[i]
        r1 = r_vals[i + 1]
        rc0 = rc_vals[i]
        rc1 = rc_vals[i + 1]

        p0 = _polar_to_cart(t0, r0)
        p1 = _polar_to_cart(t1, r1)
        q1 = _polar_to_cart(t1, rc1)
        q0 = _polar_to_cart(t0, rc0)

        p0m = _reflect_point_across_line(p0, q0, q1)
        p1m = _reflect_point_across_line(p1, q0, q1)

        t0p, r0p = _cart_to_polar(*p0)
        t1p, r1p = _cart_to_polar(*p1)
        t1q, r1q = _cart_to_polar(*q1)
        t0q, r0q = _cart_to_polar(*q0)
        units_primary.append(([t0p, t1p, t1q, t0q], [r0p, r1p, r1q, r0q]))

        t0m, r0m = _cart_to_polar(*p0m)
        t1m, r1m = _cart_to_polar(*p1m)
        units_mirror.append(([t0q, t1q, t1m, t0m], [r0q, r1q, r1m, r0m]))

    return theta_vals, r_vals, rc_vals, units_primary, units_mirror, len(units_primary)


def _rotate_points(points: List[Tuple[float, float]], angle: float) -> List[Tuple[float, float]]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]


def _build_unfold_polygons(
    a: float,
    b: float,
    dtheta_deg: int,
    p: float,
    unit_count: int,
) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    dtheta = math.radians(max(1, int(dtheta_deg)))
    gamma = math.exp(b * dtheta)

    eb = math.exp(2.0 * math.pi * b)
    c_factor = p + (1.0 - p) * eb

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

    base = _rotate_points([p0, p1, q1, q0], angle)

    primary_polys: List[List[Tuple[float, float]]] = []
    mirror_polys: List[List[Tuple[float, float]]] = []

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


def _line_segment_intersection(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
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


def interactive_forward_polar(
    a: float = 1.0,
    b: float = 0.2,
    dtheta_deg: int = 30,
    turns: float = 2.0,
    p: float = 0.5,
    elastic_percent: float = 0.05,
    elastic_enabled: bool = True,
) -> None:
    """Interactive forward design plot in polar coordinates with sliders."""
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.9],
        left=0.08,
        right=0.62,
        bottom=0.08,
        top=0.90,
        hspace=0.22,
    )
    ax = fig.add_subplot(gs[0], projection="polar")
    ax_unfold = fig.add_subplot(gs[1])

    fig.text(0.70, 0.865, "Parameters", fontsize=10, color="#7a7f87")

    info_taper = fig.text(0.10, 0.075, "", color="#5f646b", fontsize=10)
    info_tip = fig.text(0.38, 0.075, "", color="#5f646b", fontsize=10)
    info_base = fig.text(0.10, 0.05, "", color="#5f646b", fontsize=10)
    info_length = fig.text(0.38, 0.05, "", color="#5f646b", fontsize=10)
    fig.patch.set_facecolor("#f7f7f9")
    ax.set_facecolor("#ffffff")

    ax.set_title("OpenSpiRob Design (Polar)")

    last_state: dict = {}

    def draw(
        a_val: float,
        b_val: float,
        dtheta_val: float,
        theta_max_pi: float,
        p_val: float,
        elastic_pct_val: float,
        show_elastic: bool,
    ) -> None:
        ax.clear()
        ax.set_title("OpenSpiRob Design (Polar)")
        ax_unfold.clear()
        ax_unfold.set_title("Unfolded (Cartesian)")

        (
            theta_vals,
            r_vals,
            rc_vals,
            units_primary,
            units_mirror,
            unit_count,
        ) = _build_forward_units(
            a=a_val,
            b=b_val,
            dtheta_deg=int(dtheta_val),
            turns=max(0.1, theta_max_pi / 2.0),
            p=p_val,
        )

        # Original spiral
        ax.plot(
            theta_vals,
            r_vals,
            color="#1f77b4",
            linewidth=2.0,
            label="Original Spiral",
        )
        # Central spiral (shorter by 2π)
        theta_end = max(0.0, math.pi * theta_max_pi)
        rc_end = max(0.0, theta_end - 2.0 * math.pi)
        rc_theta = [t for t in theta_vals if t <= rc_end + 1e-12]
        rc_r = rc_vals[: len(rc_theta)]
        ax.plot(
            rc_theta,
            rc_r,
            color="#ff7f0e",
            linewidth=2.0,
            label="Center Spiral",
        )
        # Discrete units as filled polygons (trapezoid + mirrored trapezoid)
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

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
        )
        ax.grid(True, alpha=0.3)

        polys_primary, polys_mirror = _build_unfold_polygons(
            a=a_val,
            b=b_val,
            dtheta_deg=int(dtheta_val),
            p=p_val,
            unit_count=unit_count,
        )
        for poly in polys_primary:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            ax_unfold.fill(
                xs,
                ys,
                color="#9ecae1",
                alpha=0.35,
                edgecolor="#6baed6",
                linewidth=0.6,
            )
        for poly in polys_mirror:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            ax_unfold.fill(
                xs,
                ys,
                color="#a1d99b",
                alpha=0.35,
                edgecolor="#74c476",
                linewidth=0.6,
            )
        ray_start = (0.0, 0.0)
        ray_upper_end = (0.0, 0.0)
        ray_lower_end = (0.0, 0.0)
        elastic_poly = None
        elastic_poly_mirror = None

        # Elastic layer geometry (computed always for stable axes)
        if polys_primary:
            eb = math.exp(2.0 * math.pi * b_val)
            c_factor = p_val + (1.0 - p_val) * eb
            l_vtip = (c_factor * a_val * math.sqrt(b_val * b_val + 1.0)) / b_val
            taper_angle = 2.0 * math.atan(
                (b_val * (eb - 1.0)) / (math.sqrt(b_val * b_val + 1.0) * (eb + 1.0))
            )
            elastic_angle = max(0.0, elastic_pct_val) * (taper_angle * 0.5)
            m = math.tan(elastic_angle) if elastic_angle != 0 else 0.0

            left_poly = polys_primary[0]
            right_poly = polys_primary[-1]
            left_edge = (left_poly[0], left_poly[3])
            right_edge = (right_poly[1], right_poly[2])

            max_poly_x = max(x for poly in polys_primary for x, _y in poly)
            ray_len = max(10.0, max_poly_x + l_vtip + 10.0)
            ray_start = (-l_vtip, 0.0)
            ray_upper_end = (-l_vtip + ray_len, m * ray_len)
            ray_lower_end = (-l_vtip + ray_len, -m * ray_len)

            if show_elastic:
                upper_left = _line_segment_intersection(
                    ray_start, ray_upper_end, left_edge[0], left_edge[1]
                )
                upper_right = _line_segment_intersection(
                    ray_start, ray_upper_end, right_edge[0], right_edge[1]
                )
                q0_left = left_edge[1]
                q1_right = right_edge[1]
                if upper_left and upper_right:
                    elastic_poly = [
                        q0_left,
                        upper_left,
                        upper_right,
                        q1_right,
                    ]
                    ax_unfold.fill(
                        [p[0] for p in elastic_poly],
                        [p[1] for p in elastic_poly],
                        color="#ff7f0e",
                        alpha=0.28,
                        edgecolor="#ff7f0e",
                        linewidth=0.9,
                        zorder=3,
                    )
                    elastic_poly_mirror = [(x, -y) for x, y in elastic_poly]
                    ax_unfold.fill(
                        [p[0] for p in elastic_poly_mirror],
                        [p[1] for p in elastic_poly_mirror],
                        color="#ff7f0e",
                        alpha=0.28,
                        edgecolor="#ff7f0e",
                        linewidth=0.9,
                        zorder=3,
                    )
                ax_unfold.plot(
                    [ray_start[0], ray_upper_end[0]],
                    [ray_start[1], ray_upper_end[1]],
                    color="#9aa0a6",
                    linewidth=0.6,
                    linestyle="-",
                    alpha=0.95,
                    zorder=5,
                )
                ax_unfold.plot(
                    [ray_start[0], ray_lower_end[0]],
                    [ray_start[1], ray_lower_end[1]],
                    color="#9aa0a6",
                    linewidth=0.6,
                    linestyle="-",
                    alpha=0.95,
                    zorder=5,
                )
                ax_unfold.scatter(
                    [ray_start[0]],
                    [ray_start[1]],
                    color="#9aa0a6",
                    s=12,
                    zorder=6,
                )
                vtip_offset = 0.12 * max(1.0, abs(ray_upper_end[1]), abs(ray_lower_end[1]))
                ax_unfold.text(
                    ray_start[0],
                    ray_start[1] + vtip_offset,
                    "vtip",
                    color="#9aa0a6",
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )

        last_state.update(
            {
                "a": a_val,
                "b": b_val,
                "dtheta_deg": dtheta_val,
                "theta_max_pi": theta_max_pi,
                "p": p_val,
                "elastic_pct": elastic_pct_val,
                "theta_vals": theta_vals,
                "r_vals": r_vals,
                "rc_vals": rc_vals,
                "polys_primary": polys_primary,
                "polys_mirror": polys_mirror,
                "units_primary": units_primary,
                "units_mirror": units_mirror,
                "ray": (ray_start, ray_upper_end, ray_lower_end),
                "show_elastic": show_elastic,
                "elastic_poly": elastic_poly,
                "elastic_poly_mirror": elastic_poly_mirror,
            }
        )
        ax_unfold.set_aspect("equal", adjustable="box")
        if polys_primary:
            min_x = min(min(x for x, _y in poly) for poly in polys_primary)
            max_x = max(max(x for x, _y in poly) for poly in polys_primary)
            max_y_primary = max(max(y for _x, y in poly) for poly in polys_primary)
            max_y_mirror = max(
                max(-y for _x, y in poly) for poly in polys_mirror
            )
            max_y = max(max_y_primary, max_y_mirror)
            max_ray_y = max(abs(ray_upper_end[1]), abs(ray_lower_end[1]))
            y_limit = max(max_y, max_ray_y) * 1.15
            pad_x = 0.05 * (max_x - min_x + 1e-6)
            ax_unfold.set_xlim(min(min_x, ray_start[0]) - pad_x, max_x + pad_x)
            ax_unfold.set_ylim(-y_limit, y_limit)
        ax_unfold.set_xlabel("x (mm)")
        ax_unfold.set_ylabel("y (mm)")
        ax_unfold.grid(True, alpha=0.2)

        theta_end = max(0.0, math.pi * theta_max_pi)
        eb = math.exp(2.0 * math.pi * b_val)
        taper_angle = 2.0 * math.atan(
            (b_val * (eb - 1.0)) / (math.sqrt(b_val * b_val + 1.0) * (eb + 1.0))
        )
        if polys_primary:
            tip_size = 2.0 * max(y for _x, y in polys_primary[0])
            base_size = 2.0 * max(y for _x, y in polys_primary[-1])
        else:
            tip_size = 0.0
            base_size = 0.0
        if polys_primary:
            length = max(x for x, _y in polys_primary[-1])
        else:
            length = 0.0

        info_taper.set_text(f"Taper Angle: {math.degrees(taper_angle):.2f}°")
        info_tip.set_text(f"Tip Size: {tip_size:.2f} mm")
        info_base.set_text(f"Base Size: {base_size:.2f} mm")
        info_length.set_text(f"Robot Length: {length:.2f} mm")

        last_state["base_size"] = base_size

    ax_a = fig.add_axes([0.70, 0.80, 0.18, 0.035], facecolor="#e9ecf1")
    ax_b = fig.add_axes([0.70, 0.74, 0.18, 0.035], facecolor="#e9ecf1")
    ax_dt = fig.add_axes([0.70, 0.68, 0.18, 0.035], facecolor="#e9ecf1")
    ax_range = fig.add_axes([0.70, 0.62, 0.18, 0.035], facecolor="#e9ecf1")
    ax_p = fig.add_axes([0.70, 0.56, 0.18, 0.035], facecolor="#e9ecf1")

    ax_elastic = fig.add_axes([0.70, 0.20, 0.18, 0.035], facecolor="#e9ecf1")
    ax_elastic_val = fig.add_axes([0.90, 0.20, 0.05, 0.035])

    ax_a_val = fig.add_axes([0.90, 0.80, 0.05, 0.035])
    ax_b_val = fig.add_axes([0.90, 0.74, 0.05, 0.035])
    ax_dt_val = fig.add_axes([0.90, 0.68, 0.05, 0.035])
    ax_range_val = fig.add_axes([0.90, 0.62, 0.05, 0.035])
    ax_p_val = fig.add_axes([0.90, 0.56, 0.05, 0.035])

    ax_toggle = fig.add_axes([0.68, 0.26, 0.07, 0.06], facecolor="#f7f7f9")
    ax_save_img = fig.add_axes([0.68, 0.05, 0.10, 0.040])
    ax_save_cad = fig.add_axes([0.80, 0.05, 0.10, 0.040])
    ax_preview_3d = fig.add_axes([0.68, 0.01, 0.22, 0.035])
    ax_toggle.set_aspect("equal", adjustable="box")

    slider_a = Slider(
        ax_a, "a (mm)", 0.1, 10.0, valinit=a, valstep=0.05, color="#4a90e2"
    )
    slider_b = Slider(ax_b, "b", 0.02, 1.0, valinit=b, valstep=0.01, color="#4a90e2")
    slider_dt = Slider(
        ax_dt,
        "Δθ (deg)",
        1,
        60,
        valinit=dtheta_deg,
        valstep=1,
        valfmt="%d°",
        color="#4a90e2",
    )
    slider_range = Slider(
        ax_range,
        "θ max (π)",
        1,
        12,
        valinit=2.0 * turns,
        valstep=0.1,
        valfmt="%.1fπ",
        color="#4a90e2",
    )
    slider_p = Slider(
        ax_p,
        "p",
        0.0,
        1.0,
        valinit=p,
        valstep=0.01,
        valfmt="%.2f",
        color="#4a90e2",
    )
    slider_elastic = Slider(
        ax_elastic,
        "Elastic %",
        0.0,
        1.0,
        valinit=elastic_percent,
        valstep=0.01,
        valfmt="%d%%",
        color="#4a90e2",
    )

    # iOS-style toggle
    ax_toggle.set_xticks([])
    ax_toggle.set_yticks([])
    for spine in ax_toggle.spines.values():
        spine.set_visible(False)
    ax_toggle.set_facecolor("#f7f7f9")
    toggle_bg = FancyBboxPatch(
        (0.12, 0.18),
        0.76,
        0.32,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.0,
        edgecolor="#c7cbd1",
        facecolor="#e6e9ee",
        transform=ax_toggle.transAxes,
    )
    toggle_knob = Circle(
        (0.30, 0.34),
        0.12,
        transform=ax_toggle.transAxes,
        facecolor="#ffffff",
        edgecolor="#c7cbd1",
        linewidth=1.0,
    )
    toggle_label = ax_toggle.text(
        1.05,
        0.36,
        "Elastic Layer",
        ha="left",
        va="center",
        fontsize=10,
        color="#7a7f87",
        transform=ax_toggle.transAxes,
    )
    ax_toggle.add_patch(toggle_bg)
    ax_toggle.add_patch(toggle_knob)
    toggle_state = {"on": elastic_enabled}

    btn_save_img = Button(ax_save_img, "Save IMG", color="#eef1f5", hovercolor="#dfe4ea")
    btn_save_cad = Button(ax_save_cad, "Export CAD", color="#eef1f5", hovercolor="#dfe4ea")
    btn_preview_3d = Button(ax_preview_3d, "3D Preview", color="#eef1f5", hovercolor="#dfe4ea")
    for btn in (btn_save_img, btn_save_cad, btn_preview_3d):
        btn.ax.patch.set_edgecolor("#c7cbd1")
        btn.ax.patch.set_linewidth(1.0)

    def _render_toggle() -> None:
        if toggle_state["on"]:
            toggle_bg.set_facecolor("#4a90e2")
            toggle_bg.set_edgecolor("#4a90e2")
            toggle_knob.center = (0.66, 0.34)
        else:
            toggle_bg.set_facecolor("#e6e9ee")
            toggle_bg.set_edgecolor("#c7cbd1")
            toggle_knob.center = (0.30, 0.34)
        fig.canvas.draw_idle()

    def _save_current_images(_event) -> None:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(out_dir, exist_ok=True)
        polar_path = os.path.join(out_dir, f"polar_{ts}.png")
        unfold_path = os.path.join(out_dir, f"unfold_{ts}.png")
        vector_path = os.path.join(out_dir, f"design_{ts}.svg")
        if not last_state:
            return

        # Polar-only figure
        polar_fig, polar_ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        polar_ax.set_title("Forward Design (Polar)")
        polar_ax.plot(
            last_state["theta_vals"],
            last_state["r_vals"],
            color="#1f77b4",
            linewidth=2.0,
            label="Original Spiral",
        )
        theta_end = max(0.0, math.pi * last_state["theta_max_pi"])
        rc_end = max(0.0, theta_end - 2.0 * math.pi)
        rc_theta = [t for t in last_state["theta_vals"] if t <= rc_end + 1e-12]
        rc_r = last_state["rc_vals"][: len(rc_theta)]
        polar_ax.plot(
            rc_theta,
            rc_r,
            color="#ff7f0e",
            linewidth=2.0,
            label="Center Spiral",
        )
        for theta_poly, r_poly in last_state["units_primary"]:
            polar_ax.fill(
                theta_poly,
                r_poly,
                color="#9ecae1",
                alpha=0.35,
                edgecolor="#6baed6",
                linewidth=0.6,
            )
        for theta_poly, r_poly in last_state["units_mirror"]:
            polar_ax.fill(
                theta_poly,
                r_poly,
                color="#a1d99b",
                alpha=0.35,
                edgecolor="#74c476",
                linewidth=0.6,
            )
        polar_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
        polar_ax.grid(True, alpha=0.3)
        polar_fig.savefig(polar_path, dpi=150, bbox_inches="tight")
        plt.close(polar_fig)

        # Unfold-only figure
        unfold_fig, unfold_ax = plt.subplots(figsize=(6, 4.8))
        unfold_ax.set_title("Unfolded (Cartesian)")
        for poly in last_state["polys_primary"]:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            unfold_ax.fill(
                xs,
                ys,
                color="#9ecae1",
                alpha=0.35,
                edgecolor="#6baed6",
                linewidth=0.6,
            )
        for poly in last_state["polys_mirror"]:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            unfold_ax.fill(
                xs,
                ys,
                color="#a1d99b",
                alpha=0.35,
                edgecolor="#74c476",
                linewidth=0.6,
            )
        if last_state["show_elastic"] and last_state["elastic_poly"]:
            elastic_poly = last_state["elastic_poly"]
            elastic_poly_mirror = last_state["elastic_poly_mirror"]
            unfold_ax.fill(
                [p[0] for p in elastic_poly],
                [p[1] for p in elastic_poly],
                color="#ff7f0e",
                alpha=0.28,
                edgecolor="#ff7f0e",
                linewidth=0.9,
                zorder=3,
            )
            if elastic_poly_mirror:
                unfold_ax.fill(
                    [p[0] for p in elastic_poly_mirror],
                    [p[1] for p in elastic_poly_mirror],
                    color="#ff7f0e",
                    alpha=0.28,
                    edgecolor="#ff7f0e",
                    linewidth=0.9,
                    zorder=3,
                )
        ray_start, ray_upper_end, ray_lower_end = last_state["ray"]
        if last_state["show_elastic"]:
            unfold_ax.plot(
                [ray_start[0], ray_upper_end[0]],
                [ray_start[1], ray_upper_end[1]],
                color="#9aa0a6",
                linewidth=0.6,
                alpha=0.95,
            )
            unfold_ax.plot(
                [ray_start[0], ray_lower_end[0]],
                [ray_start[1], ray_lower_end[1]],
                color="#9aa0a6",
                linewidth=0.6,
                alpha=0.95,
            )
            unfold_ax.scatter([ray_start[0]], [ray_start[1]], color="#9aa0a6", s=12)
            vtip_offset = 0.12 * max(1.0, abs(ray_upper_end[1]), abs(ray_lower_end[1]))
            unfold_ax.text(
                ray_start[0],
                ray_start[1] + vtip_offset,
                "vtip",
                color="#9aa0a6",
                fontsize=8,
                va="bottom",
                ha="left",
            )
        unfold_ax.set_aspect("equal", adjustable="box")
        if last_state["polys_primary"]:
            min_x = min(min(x for x, _y in poly) for poly in last_state["polys_primary"])
            max_x = max(max(x for x, _y in poly) for poly in last_state["polys_primary"])
            max_y_primary = max(
                max(y for _x, y in poly) for poly in last_state["polys_primary"]
            )
            max_y_mirror = max(
                max(-y for _x, y in poly) for poly in last_state["polys_mirror"]
            )
            max_y = max(max_y_primary, max_y_mirror)
            max_ray_y = max(abs(ray_upper_end[1]), abs(ray_lower_end[1]))
            y_limit = max(max_y, max_ray_y) * 1.15
            pad_x = 0.05 * (max_x - min_x + 1e-6)
            unfold_ax.set_xlim(min(min_x, ray_start[0]) - pad_x, max_x + pad_x)
            unfold_ax.set_ylim(-y_limit, y_limit)
        unfold_ax.set_xlabel("x (mm)")
        unfold_ax.set_ylabel("y (mm)")
        unfold_ax.grid(True, alpha=0.2)
        unfold_fig.savefig(unfold_path, dpi=150, bbox_inches="tight")
        plt.close(unfold_fig)

        # Full vector export with parameters
        param_text = (
            f"a={last_state['a']:.4f} mm, b={last_state['b']:.4f}, "
            f"Δθ={int(last_state['dtheta_deg'])}°, θmax={last_state['theta_max_pi']:.1f}π, "
            f"p={last_state['p']:.2f}, elastic={last_state['elastic_pct']*100:.0f}%"
        )
        label = fig.text(
            0.02,
            0.965,
            param_text,
            fontsize=9,
            color="#5f646b",
        )
        fig.savefig(vector_path, format="svg", bbox_inches="tight")
        label.remove()

    def _save_cad_placeholder(_event) -> None:
        if not last_state:
            return
        out_dir = os.path.join(os.getcwd(), "exports")
        polygons = []
        polygons.extend(last_state.get("polys_primary", []))
        polygons.extend(last_state.get("polys_mirror", []))
        if last_state.get("show_elastic"):
            if last_state.get("elastic_poly"):
                polygons.append(last_state["elastic_poly"])
            if last_state.get("elastic_poly_mirror"):
                polygons.append(last_state["elastic_poly_mirror"])
        base_size = last_state.get("base_size", 0.0)
        thickness = max(0.1, base_size * 0.5)
        try:
            from .export_cad import export_cad

            export_cad(polygons, thickness=thickness, out_dir=out_dir, prefix="spi_rob")
            print(f"[Export CAD] Saved to: {out_dir}")
        except ModuleNotFoundError:
            print("[Export CAD] cadquery not found. Install it in this venv:")
            print("  python -m pip install cadquery")
        except Exception as exc:
            print(f"[Export CAD] Failed: {exc}")

    preview_state = {"on": False, "fig": None, "ax": None}

    def _render_3d_preview() -> None:
        if not preview_state["on"] or not last_state:
            return
        try:
            from .export_cad import build_solid
        except Exception as exc:
            print(f"[3D Preview] Failed to import CAD builder: {exc}")
            return
        polygons = []
        polygons.extend(last_state.get("polys_primary", []))
        polygons.extend(last_state.get("polys_mirror", []))
        if last_state.get("show_elastic"):
            if last_state.get("elastic_poly"):
                polygons.append(last_state["elastic_poly"])
            if last_state.get("elastic_poly_mirror"):
                polygons.append(last_state["elastic_poly_mirror"])
        base_size = last_state.get("base_size", 0.0)
        thickness = max(0.1, base_size * 0.5)
        try:
            solid = build_solid(polygons, thickness)
            shape = solid.val()
            verts, tris = shape.tessellate(0.5)
        except Exception as exc:
            print(f"[3D Preview] Tessellation failed: {exc}")
            return

        ax3d = preview_state["ax"]
        ax3d.clear()
        faces = []
        for tri in tris:
            try:
                idx = list(tri)
            except Exception:
                idx = [tri[0], tri[1], tri[2]]
            def _v(i: int):
                v = verts[i]
                if hasattr(v, "toTuple"):
                    return v.toTuple()
                if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
                    return (v.x, v.y, v.z)
                return (v[0], v[1], v[2])

            face = [_v(idx[0]), _v(idx[1]), _v(idx[2])]
            faces.append(face)
        mesh = Poly3DCollection(faces, facecolor="#8fbce6", edgecolor="#5b8dbd", linewidths=0.2, alpha=0.9)
        ax3d.add_collection3d(mesh)
        def _v_tuple(v):
            if hasattr(v, "toTuple"):
                return v.toTuple()
            if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
                return (v.x, v.y, v.z)
            return (v[0], v[1], v[2])

        vts = [_v_tuple(v) for v in verts]
        xs = [v[0] for v in vts]
        ys = [v[1] for v in vts]
        zs = [v[2] for v in vts]
        ax3d.set_xlim(min(xs), max(xs))
        ax3d.set_ylim(min(ys), max(ys))
        ax3d.set_zlim(min(zs), max(zs))
        ax3d.set_box_aspect((max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)))
        ax3d.set_title("3D Preview", fontsize=10)
        preview_state["fig"].canvas.draw_idle()

    def _toggle_3d_preview(_event) -> None:
        if not preview_state["on"]:
            fig3d = plt.figure(figsize=(6, 4.5))
            ax3d = fig3d.add_subplot(111, projection="3d")
            try:
                fig3d.show()
            except Exception:
                pass
            try:
                fig3d.canvas.manager.window.lift()
                fig3d.canvas.manager.window.attributes("-topmost", 1)
                fig3d.canvas.manager.window.attributes("-topmost", 0)
            except Exception:
                pass
            preview_state.update({"on": True, "fig": fig3d, "ax": ax3d})
            _render_3d_preview()
        else:
            try:
                plt.close(preview_state["fig"])
            except Exception:
                pass
            preview_state.update({"on": False, "fig": None, "ax": None})

    def _close_all(_event) -> None:
        try:
            if preview_state["fig"] is not None:
                plt.close(preview_state["fig"])
        finally:
            plt.close("all")

    for ax_slider in (ax_a, ax_b, ax_dt, ax_range, ax_p, ax_elastic):
        for spine in ax_slider.spines.values():
            spine.set_visible(False)
        ax_slider.set_xticks([])
        ax_slider.set_yticks([])
        rounded = FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax_slider.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=0.0,
            facecolor="#e9ecf1",
            zorder=-1,
        )
        ax_slider.add_patch(rounded)

    for slider in (slider_a, slider_b, slider_dt, slider_range, slider_p, slider_elastic):
        slider.track.set_facecolor("#e9ecf1")
        slider.vline.set_color("#4a90e2")
        handle = getattr(slider, "handle", getattr(slider, "_handle", None))
        if handle is not None:
            if hasattr(handle, "set_edgecolor"):
                handle.set_edgecolor("#ffffff")
            if hasattr(handle, "set_facecolor"):
                handle.set_facecolor("#4a90e2")
            if hasattr(handle, "set_markeredgecolor"):
                handle.set_markeredgecolor("#ffffff")
            if hasattr(handle, "set_markerfacecolor"):
                handle.set_markerfacecolor("#4a90e2")
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(1.0)
            elif hasattr(handle, "set_markersize"):
                handle.set_markersize(8)
        slider.label.set_color("#7a7f87")
        slider.valtext.set_color("#7a7f87")
        slider.valtext.set_visible(False)

    def _style_textbox(tb: TextBox) -> None:
        for spine in tb.ax.spines.values():
            spine.set_visible(False)
        tb.ax.set_facecolor("#eef1f5")
        tb.text_disp.set_color("#5f646b")
        tb.text_disp.set_fontsize(9)

    tb_a = TextBox(ax_a_val, "", initial=f"{a:.4f}")
    tb_b = TextBox(ax_b_val, "", initial=f"{b:.4f}")
    tb_dt = TextBox(ax_dt_val, "", initial=f"{int(dtheta_deg)}")
    tb_range = TextBox(ax_range_val, "", initial=f"{2.0 * turns:.1f}")
    tb_p = TextBox(ax_p_val, "", initial=f"{p:.2f}")
    tb_elastic = TextBox(ax_elastic_val, "", initial=f"{elastic_percent * 100.0:.0f}")

    for tb in (tb_a, tb_b, tb_dt, tb_range, tb_p, tb_elastic):
        _style_textbox(tb)

    def update(_val: float) -> None:
        show_elastic = toggle_state["on"]
        draw(
            slider_a.val,
            slider_b.val,
            slider_dt.val,
            slider_range.val,
            slider_p.val,
            slider_elastic.val,
            show_elastic,
        )
        tb_a.set_val(f"{slider_a.val:.4f}")
        tb_b.set_val(f"{slider_b.val:.4f}")
        tb_dt.set_val(f"{int(slider_dt.val)}")
        tb_range.set_val(f"{slider_range.val:.1f}")
        tb_p.set_val(f"{slider_p.val:.2f}")
        if show_elastic:
            tb_elastic.set_val(f"{slider_elastic.val * 100.0:.0f}")
        fig.canvas.draw_idle()
        _render_3d_preview()

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_dt.on_changed(update)
    slider_range.on_changed(update)
    slider_p.on_changed(update)
    slider_elastic.on_changed(update)

    def _clamp(val: float, vmin: float, vmax: float) -> float:
        return max(vmin, min(vmax, val))

    def _on_textbox(tb: TextBox, slider: Slider, is_int: bool = False) -> None:
        try:
            value = float(tb.text.strip())
        except ValueError:
            return
        if is_int:
            value = float(int(round(value)))
        value = _clamp(value, slider.valmin, slider.valmax)
        slider.set_val(value)

    tb_a.on_submit(lambda _t: _on_textbox(tb_a, slider_a, is_int=False))
    tb_b.on_submit(lambda _t: _on_textbox(tb_b, slider_b, is_int=False))
    tb_dt.on_submit(lambda _t: _on_textbox(tb_dt, slider_dt, is_int=True))
    tb_range.on_submit(lambda _t: _on_textbox(tb_range, slider_range, is_int=False))
    tb_p.on_submit(lambda _t: _on_textbox(tb_p, slider_p, is_int=False))
    def _on_percent_textbox(tb: TextBox, slider: Slider) -> None:
        try:
            value = float(tb.text.strip())
        except ValueError:
            return
        value = _clamp(value, 0.0, 100.0)
        slider.set_val(value / 100.0)

    tb_elastic.on_submit(lambda _t: _on_percent_textbox(tb_elastic, slider_elastic))

    def _set_elastic_visible(visible: bool) -> None:
        ax_elastic.set_visible(visible)
        ax_elastic_val.set_visible(visible)
        tb_elastic.ax.set_visible(visible)

    _set_elastic_visible(elastic_enabled)
    _render_toggle()

    def _toggle_event(event) -> None:
        if event.inaxes != ax_toggle:
            return
        toggle_state["on"] = not toggle_state["on"]
        _set_elastic_visible(toggle_state["on"])
        _render_toggle()
        update(0)

    fig.canvas.mpl_connect("button_press_event", _toggle_event)
    fig.canvas.mpl_connect("close_event", _close_all)
    btn_save_img.on_clicked(_save_current_images)
    btn_save_cad.on_clicked(_save_cad_placeholder)
    btn_preview_3d.on_clicked(_toggle_3d_preview)

    draw(
        a,
        b,
        dtheta_deg,
        slider_range.val,
        slider_p.val,
        slider_elastic.val,
        elastic_enabled,
    )
    plt.show()

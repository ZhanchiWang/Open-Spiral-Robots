from __future__ import annotations

import math
from pathlib import Path


def generate_mujoco_xml(
    output_path: str | Path,
    stl_name: str,
    unit_height: float,
    scale: float,
    num_units: int,
    joint_type: str,
    joint_limit_deg: float | None = None,
    robot_length: float | None = None,
    site_points: tuple[float, float, float, float] | None = None,
    cable_mode: int = 2,
    joint_stiffness: float = 0.04,
    joint_damping: float = 0.02,
    gravity: float = 0.0,
    joint_angles: list[float] | None = None,
    bend_dir_deg: float = 0.0,
    extrusion_depth: float = 0.0,
    tendon_configs: list[dict] | None = None,
) -> None:
    """Generate a MuJoCo XML chain using a single STL mesh.

    joint_angles: per-joint bend angles in degrees (length = num_units - 1).
                  Each angle rotates the child body about the joint axis.
    bend_dir_deg: bend direction for ball joints (degrees).
    """
    output_path = Path(output_path)
    num_units = max(1, int(num_units))
    joint_type = joint_type.lower().strip()
    if joint_type not in {"hinge", "ball"}:
        joint_type = "hinge"

    # Scale shrinks each unit by 1/scale (per requirement).
    # Build meshes from largest (base / link1) to smallest (tip / linkN).
    scale_step = 1.0 / max(scale, 1e-9)
    unit_scale = 0.001  # mm -> m for MuJoCo

    n_cables = max(2, int(cable_mode))

    # Per-tendon config: ctrl range, gear, and the joint (1-based) the cable
    # connects to. joint = num_units-1 (or unset) means the cable runs from the
    # root to the tip (links 1..N) as before. A smaller joint j means the cable
    # only spans links 1..j+1, so it drives joints 1..j.
    def _tendon_plan(k: int) -> dict:
        cfg: dict = {}
        if tendon_configs is not None and k < len(tendon_configs) and tendon_configs[k]:
            cfg = tendon_configs[k]
        joint = int(cfg.get("joint", num_units - 1))
        joint = max(0, min(joint, num_units - 1))
        end_link = max(1, min(joint + 1, num_units))
        return {
            "gear": float(cfg.get("gear", 50.0)),
            "ctrl_lo": float(cfg.get("ctrl_lo", -1.0)),
            "ctrl_hi": float(cfg.get("ctrl_hi", 1.0)),
            "joint": joint,
            "end_link": end_link,
        }

    tend_plans = [_tendon_plan(k) for k in range(n_cables)]

    current_scale = 1.0
    meshes: list[str] = []
    for i in range(num_units):
        mesh_name = f"baselink_{i+1}"
        meshes.append(
            f'    <mesh name="{mesh_name}" file="{stl_name}" scale="{current_scale * unit_scale:.9f} {current_scale * unit_scale:.9f} {current_scale * unit_scale:.9f}"/>'
        )
        current_scale *= scale_step  # shrink towards tip

    def joint_xml(stiffness: float, damping: float) -> str:
        if joint_type == "hinge":
            if joint_limit_deg is None:
                return (
                    f'<joint type="hinge" axis="1 0 0" stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'
                )
            limit_rad = math.radians(abs(joint_limit_deg))
            return (
                f'<joint type="hinge" axis="1 0 0" range="{-limit_rad:.9f} {limit_rad:.9f}" '
                f'stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'
            )
        # ball joints get the same ±Δθ limit as hinges (paper, Note S3): a
        # cone of half-angle Δθ around the rest orientation
        if joint_limit_deg is None:
            return f'<joint type="ball" stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'
        cone_rad = math.radians(abs(joint_limit_deg))
        return (
            f'<joint type="ball" range="0 {cone_rad:.9f}" '
            f'stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'
        )

    def build_sites(scale_factor: float, link_index: int) -> list[str]:
        if site_points is None or robot_length is None:
            return []
        x1, y1, x2, y2 = site_points
        z1 = robot_length - x1
        z2 = robot_length - x2
        n = max(2, int(cable_mode))
        sites_mm: list[tuple[float, float, float]] = []
        for k in range(n):
            # Cable 1 points +y; the rest are evenly spaced (regular polygon)
            ang = math.radians(90.0 + k * 360.0 / n)
            ca, sa = math.cos(ang), math.sin(ang)
            if abs(ca) < 1e-12:
                ca = 0.0
            if abs(sa) < 1e-12:
                sa = 0.0
            sites_mm.append((y2 * ca, y2 * sa, z2))
            sites_mm.append((y1 * ca, y1 * sa, z1))
        size_m = 0.001 * scale_factor
        sites_xml: list[str] = []
        for idx, (sx, sy, sz) in enumerate(sites_mm, start=1):
            px = sx * scale_factor * unit_scale
            py = sy * scale_factor * unit_scale
            pz = sz * scale_factor * unit_scale
            sites_xml.append(
                f'<site name="s{link_index}_{idx}" pos="{px:.9f} {py:.9f} {pz:.9f}" size="{size_m:.6f} {size_m:.6f} {size_m:.6f}"/>'
            )
        return sites_xml

    # Build nested bodies (kinematic tree)
    # link1 = base (largest), linkN = tip (smallest)
    current_scale = 1.0

    lines: list[str] = []
    lines.append('<mujoco model="spiral_robot">')
    lines.append('  <compiler angle="radian"/>')
    _grav = float(gravity)
    _gvec = "0 0 0" if _grav == 0 else f"0 0 {-_grav:.9g}"
    lines.append(f'  <option gravity="{_gvec}"/>')
    lines.append('  <asset>')
    lines.extend(meshes)
    lines.append(
        '    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>'
    )
    lines.append(
        '    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>'
    )
    lines.append('  </asset>')
    lines.append('  <worldbody>')
    lines.append('    <geom name="ground" type="plane" size="10 10 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>')
    lines.append(
        '    <light pos="1 0.5 0.7" dir="-1 -0.5 -1" directional="true" castshadow="false" diffuse="1 1 1" specular="0.3 0.3 0.3"/>'
    )

    joint_stiffness = float(joint_stiffness)
    joint_damping = float(joint_damping)

    # Build per-joint quaternion for pre-bent pose.
    # Angles arrive in base-to-tip order: J1 = base joint, J(N-1) = tip joint.
    joint_angles_rad: list[float] = []
    if joint_angles is not None:
        # The CAD pose (_apply_joint_pose) turns the base side by +angle with
        # the tip held still, so the child link turns by -angle relative to
        # its parent; the same sign keeps the XML pose equal to the CAD/STL.
        joint_angles_rad = [-math.radians(a) for a in joint_angles[:num_units - 1]]
    # Pad to num_units-1 with zeros
    while len(joint_angles_rad) < num_units - 1:
        joint_angles_rad.append(0.0)

    if joint_type == "hinge":
        # rotation about X axis in XML coordinates
        joint_axis = (1.0, 0.0, 0.0)
    else:
        # ball joint: rotation about (cos(β), -sin(β), 0) in XML coords
        beta = math.radians(bend_dir_deg)
        joint_axis = (math.cos(beta), -math.sin(beta), 0.0)

    def _quat_str(angle_rad: float, axis: tuple[float, float, float] | None = None) -> str:
        """MuJoCo wxyz quaternion for rotation about `axis` (or `joint_axis`) by angle_rad."""
        ax = axis if axis is not None else joint_axis
        if abs(angle_rad) < 1e-9:
            return ""
        half = angle_rad / 2.0
        w = math.cos(half)
        x = ax[0] * math.sin(half)
        y = ax[1] * math.sin(half)
        z = ax[2] * math.sin(half)
        return f' quat="{w:.9f} {x:.9f} {y:.9f} {z:.9f}"'

    # Root body: keep the chain vertical so the robot stands upright. The
    # exported STL already has the chain along +Z with the base face at its
    # local origin, and the nested bodies stack along +Z, so no root rotation
    # is applied. link1 (the largest = base) is a direct child of worldbody
    # with no joint, so it is welded to the ground: the root is fixed.
    # root_z = 0 puts the base face (mesh local origin) exactly on the ground.
    root_z = 0.0
    root_quat_attr = ""

    lines.append(f'    <body name="link1" pos="0 0 {root_z:.9f}"{root_quat_attr}>')
    lines.append('      <geom type="mesh" mesh="baselink_1"/>')
    for site in build_sites(1.0, 1):
        lines.append(f'      {site}')

    for i in range(1, num_units):
        z_offset = unit_height * current_scale * unit_scale
        link_idx = i + 1
        quat_attr = _quat_str(joint_angles_rad[i - 1])
        lines.append(f'      <body name="link{link_idx}" pos="0 0 {z_offset:.9f}"{quat_attr}>')
        lines.append(f'        {joint_xml(joint_stiffness, joint_damping)}')
        lines.append(f'        <geom type="mesh" mesh="baselink_{i+1}"/>')
        # child is smaller than parent: multiply to shrink towards tip
        child_scale = current_scale * scale_step
        for site in build_sites(child_scale, link_idx):
            lines.append(f'        {site}')
        current_scale *= scale_step
        # stiffness and damping decrease towards tip (smaller units)
        joint_stiffness *= scale_step * scale_step
        joint_damping *= scale_step

    for _ in range(num_units - 1):
        lines.append('      </body>')
    lines.append('    </body>')
    lines.append('  </worldbody>')

    # tendons
    tendon_lengths: dict[str, float] = {}
    if num_units > 0 and site_points is not None and robot_length is not None:
        # Precompute site positions per link for length calculation (meters).
        # The <site> positions written to the XML are each body's LOCAL,
        # already-scaled coordinates. The tendon length MuJoCo actually sees is
        # the serpentine path through those sites in the WORLD frame, where
        # every link is translated along its parent's +Z by
        # unit_height * current_scale * unit_scale and rotated by its bend
        # quaternion. The old code summed the local coords directly (ignoring
        # those transforms) AND chained the sites in the wrong order, producing a
        # meaningless value that fell back to robot_length * unit_scale * 2 — a
        # range far below the real tendon. Compute the true world-frame length
        # so the generated <spatial> range brackets the actual tendon.
        local_sites: dict[int, list[tuple[float, float, float]]] = {}
        sc = 1.0
        for link_idx in range(1, num_units + 1):
            pts: list[tuple[float, float, float]] = []
            for site in build_sites(sc, link_idx):
                try:
                    pos_str = site.split('pos="', 1)[1].split('"', 1)[0]
                    pts.append(tuple(float(v) for v in pos_str.split()))
                except Exception:
                    continue
            local_sites[link_idx] = pts
            # scale shrinks towards tip (linkN = smallest)
            sc *= scale_step

        def _rotate(q, v):
            """Rotate vector v (3) by unit quaternion q (wxyz)."""
            w, qx, qy, qz = q
            vx, vy, vz = v
            tx = 2.0 * (qy * vz - qz * vy)
            ty = 2.0 * (qz * vx - qx * vz)
            tz = 2.0 * (qx * vy - qy * vx)
            return (
                vx + w * tx + (qy * tz - qz * ty),
                vy + w * ty + (qz * tx - qx * tz),
                vz + w * tz + (qx * ty - qy * tx),
            )

        def _qmul(a, b):
            """Hamilton quaternion product (wxyz)."""
            w1, x1, y1, z1 = a
            w2, x2, y2, z2 = b
            return (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )

        def _bend_quat(angle_rad: float):
            """wxyz quaternion for the per-link bend about the joint axis."""
            half = angle_rad / 2.0
            s = math.sin(half)
            return (math.cos(half), joint_axis[0] * s, joint_axis[1] * s, joint_axis[2] * s)

        # World-frame site positions per link, using the same scalar progression
        # as the body-building loop above. Each site is translated by the
        # accumulated body position AND rotated by the accumulated bend quat.
        world_sites: dict[int, list[tuple[float, float, float]]] = {}
        world_pos = (0.0, 0.0, 0.0)
        world_quat = (1.0, 0.0, 0.0, 0.0)
        sc = 1.0

        def _world_pts(local_pts):
            """local points -> world: world_pos + world_quat applied."""
            out_pts = []
            for p in local_pts:
                r = _rotate(world_quat, p)
                out_pts.append((world_pos[0] + r[0], world_pos[1] + r[1], world_pos[2] + r[2]))
            return out_pts

        world_sites[1] = _world_pts(local_sites.get(1, []))
        for i in range(1, num_units):
            z_offset = unit_height * sc * unit_scale
            _tr = _rotate(world_quat, (0.0, 0.0, z_offset))
            world_pos = (world_pos[0] + _tr[0], world_pos[1] + _tr[1], world_pos[2] + _tr[2])
            world_quat = _qmul(world_quat, _bend_quat(joint_angles_rad[i - 1]))
            link_idx = i + 1
            # sites of this link live in its own (translated + rotated) frame
            world_sites[link_idx] = _world_pts(local_sites.get(link_idx, []))
            sc *= scale_step

        def _tendon_length_serpentine(k: int, end_link: int) -> float:
            """World-frame length of cable k's serpentine path: for each covered
            link its outer (2k+1) then inner (2k+2) site, in the same order the
            <spatial> tendon lists them."""
            pts: list[tuple[float, float, float]] = []
            for i in range(1, end_link + 1):
                arr = world_sites.get(i, [])
                for site_idx in (2 * k + 1, 2 * k + 2):
                    if site_idx - 1 < 0 or site_idx - 1 >= len(arr):
                        continue
                    pts.append(arr[site_idx - 1])
            total = 0.0
            for a, b in zip(pts, pts[1:]):
                total += math.dist(a, b)
            return total

        for k in range(n_cables):
            end_link = tend_plans[k]["end_link"]
            tendon_lengths[f"tendon{k+1}"] = _tendon_length_serpentine(k, end_link)

    if num_units > 0:
        lines.append('  <tendon>')
        tendon_ranges: list[tuple[float, float]] = []
        for k in range(n_cables):
            plan = tend_plans[k]
            end_link = plan["end_link"]
            # Use actual computed tendon length to set a safe range
            t_len = tendon_lengths.get(f"tendon{k+1}", 0.0)
            if t_len <= 0.0 and robot_length is not None:
                t_len = robot_length * unit_scale * 2.0  # fallback
            lo = max(0.001, t_len * 0.3)
            hi = max(lo + 0.001, t_len * 3.0)
            lr = (lo, hi)
            tendon_ranges.append(lr)
            lines.append(f'    <spatial name="tendon{k+1}" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{lr[0]:.9f} {lr[1]:.9f}">')
            for i in range(1, end_link + 1):
                lines.append(f'      <site site="s{i}_{2 * k + 1}"/>')
                lines.append(f'      <site site="s{i}_{2 * k + 2}"/>')
            lines.append('    </spatial>')
        lines.append('  </tendon>')

        lines.append('  <actuator>')
        for k in range(n_cables):
            plan = tend_plans[k]
            # motor gives direct force control (no activation dynamics).
            # ctrl rises from ctrl_lo to ctrl_hi: positive = shorten tendon.
            lines.append(
                f'    <motor name="act{k+1}" tendon="tendon{k+1}" ctrllimited="true" '
                f'ctrlrange="{plan["ctrl_lo"]:.9g} {plan["ctrl_hi"]:.9g}" gear="{plan["gear"]:.9g}"/>'
            )
        lines.append('  </actuator>')
    lines.append('</mujoco>')

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

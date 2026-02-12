import math
from dataclasses import dataclass
from typing import Optional

from .spiral import SpiralSamples, sample_spiral


@dataclass(frozen=True)
class DesignResult:
    a: float
    b: float
    theta_start: float
    theta_end: float
    samples: SpiralSamples


def forward_design(
    a: float,
    b: float,
    dtheta: float,
    theta_start: float = 0.0,
    theta_end: Optional[float] = None,
    turns: Optional[float] = 2.0,
) -> DesignResult:
    """
    Forward design: sample a logarithmic spiral from parameters.

    Provide either theta_end directly or turns (default 2 turns).
    """
    if theta_end is None:
        if turns is None:
            raise ValueError("Provide theta_end or turns.")
        theta_end = theta_start + 2.0 * math.pi * turns

    samples = sample_spiral(
        a=a,
        b=b,
        theta_start=theta_start,
        theta_end=theta_end,
        dtheta=dtheta,
    )
    return DesignResult(
        a=a,
        b=b,
        theta_start=theta_start,
        theta_end=theta_end,
        samples=samples,
    )


def _solve_b_for_length(r0: float, r1: float, length: float) -> float:
    """
    Solve for b in:
      length = (sqrt(1 + b^2) / b) * (r1 - r0)
    for b > 0 using bisection.
    """
    if r1 <= r0:
        raise ValueError("r1 must be > r0.")
    min_length = r1 - r0
    if length < min_length:
        raise ValueError(
            "length is too small for the given radii; must be >= r1 - r0."
        )

    def f(b: float) -> float:
        return (math.sqrt(1.0 + b * b) / b) * (r1 - r0) - length

    low = 1e-6
    high = 1000.0
    while f(high) > 0:
        high *= 2.0
        if high > 1e9:
            raise ValueError("Failed to bracket a solution for b.")

    for _ in range(80):
        mid = 0.5 * (low + high)
        if f(mid) > 0:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def _solve_b_from_widths_and_length(
    width_root: float,
    width_tip: float,
    length: float,
) -> float:
    """
    Solve b from paper-derived relation:

    L = (sqrt(b^2+1)/b) * (width_root - width_tip) * 0.5 * (e^(2πb)+1)/(e^(2πb)-1)
    """
    if width_root <= width_tip:
        raise ValueError("width_root must be > width_tip.")
    if length <= 0:
        raise ValueError("length must be > 0.")

    def f(b: float) -> float:
        if b <= 0:
            return float("inf")
        eb = math.exp(2.0 * math.pi * b)
        ratio = (eb + 1.0) / (eb - 1.0)
        return (math.sqrt(1.0 + b * b) / b) * (width_root - width_tip) * 0.5 * ratio - length

    low = 1e-6
    high = 5.0
    while f(high) > 0:
        high *= 2.0
        if high > 1e6:
            raise ValueError("Failed to bracket a solution for b.")

    for _ in range(80):
        mid = 0.5 * (low + high)
        if f(mid) > 0:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def inverse_design_paper(
    width_root: float,
    width_tip: float,
    length: float,
    dtheta: float,
    theta_start: float = 0.0,
) -> DesignResult:
    """
    Inverse design using SpiRobs paper relations (Eqs. 1-5).

    width_tip = d(0) = a * (e^(2πb) - 1)
    width_root = d(q0) = a * e^(b q0) * (e^(2πb) - 1)
    L(0; q0) = (sqrt(b^2+1)/b) * (a/2) * (e^(2πb)+1) * (e^(b q0) - 1)
    """
    if width_root <= 0 or width_tip <= 0 or length <= 0:
        raise ValueError("widths and length must be > 0.")
    if width_root <= width_tip:
        raise ValueError("width_root must be > width_tip.")

    b = _solve_b_from_widths_and_length(width_root, width_tip, length)
    eb = math.exp(2.0 * math.pi * b)
    a = width_tip / (eb - 1.0)
    q0 = math.log(width_root / width_tip) / b

    theta_end = theta_start + q0

    samples = sample_spiral(
        a=a,
        b=b,
        theta_start=theta_start,
        theta_end=theta_end,
        dtheta=dtheta,
    )
    return DesignResult(
        a=a,
        b=b,
        theta_start=theta_start,
        theta_end=theta_end,
        samples=samples,
    )


def inverse_design(
    width_root: float,
    width_tip: float,
    length: float,
    dtheta: float,
    theta_start: float = 0.0,
    method: str = "paper",
    width_to_radius: float = 0.5,
) -> DesignResult:
    """
    Inverse design wrapper.

    method="paper" uses the SpiRobs equations (recommended).
    method="basic" uses a simple radius~width proportionality assumption.
    """
    method = method.lower().strip()
    if method == "paper":
        return inverse_design_paper(
            width_root=width_root,
            width_tip=width_tip,
            length=length,
            dtheta=dtheta,
            theta_start=theta_start,
        )
    if method != "basic":
        raise ValueError("method must be 'paper' or 'basic'.")

    if width_root <= 0 or width_tip <= 0 or length <= 0:
        raise ValueError("widths and length must be > 0.")
    if width_root <= width_tip:
        raise ValueError("width_root must be > width_tip.")
    if width_to_radius <= 0:
        raise ValueError("width_to_radius must be > 0.")

    r0 = width_to_radius * width_tip
    r1 = width_to_radius * width_root

    b = _solve_b_for_length(r0, r1, length)
    a = r0
    theta_end = theta_start + math.log(r1 / r0) / b

    samples = sample_spiral(
        a=a,
        b=b,
        theta_start=theta_start,
        theta_end=theta_end,
        dtheta=dtheta,
    )
    return DesignResult(
        a=a,
        b=b,
        theta_start=theta_start,
        theta_end=theta_end,
        samples=samples,
    )

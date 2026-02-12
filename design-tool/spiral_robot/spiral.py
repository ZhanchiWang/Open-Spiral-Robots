import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class SpiralSamples:
    theta: List[float]
    r: List[float]
    x: List[float]
    y: List[float]


def log_spiral_r(theta: float, a: float, b: float) -> float:
    """Return r for a logarithmic spiral r = a * exp(b * theta)."""
    return a * math.exp(b * theta)


def sample_spiral(
    a: float,
    b: float,
    theta_start: float,
    theta_end: float,
    dtheta: float,
) -> SpiralSamples:
    """Sample a logarithmic spiral in polar and Cartesian coordinates."""
    if dtheta <= 0:
        raise ValueError("dtheta must be > 0.")
    if theta_end <= theta_start:
        raise ValueError("theta_end must be > theta_start.")

    theta_values: List[float] = []
    r_values: List[float] = []
    x_values: List[float] = []
    y_values: List[float] = []

    theta = theta_start
    while theta <= theta_end + 1e-12:
        r = log_spiral_r(theta, a, b)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        theta_values.append(theta)
        r_values.append(r)
        x_values.append(x)
        y_values.append(y)
        theta += dtheta

    return SpiralSamples(
        theta=theta_values,
        r=r_values,
        x=x_values,
        y=y_values,
    )

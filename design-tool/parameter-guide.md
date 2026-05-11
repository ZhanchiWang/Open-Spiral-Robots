# OpenSpiRob Parameter Guide

Use this reference to understand how each parameter affects the spiral robot geometry and exported design.

---

## Core Geometry Parameters

### a (mm)
**Initial radius of the logarithmic spiral.**

- Controls the starting size of the spiral tip.
- Higher values create a wider tip.
- Lower values create a sharper and smaller front section.

**Typical range:** `0.5 – 5.0`

---

### b
**Spiral growth rate.**

- Determines how fast the spiral expands as angle increases.
- Low values = gradual expansion.
- High values = aggressive widening.

**Effect on robot:**

- Larger body diameter
- Faster taper change
- Shorter compact geometry

**Typical range:** `0.08 – 0.30`

---

### Δθ (deg)
**Angular step between generated segments.**

- Defines discretization resolution.
- Lower values create more segments.
- Higher values create fewer, larger segments.

**Effect on robot:**

- Smaller Δθ = smoother geometry
- Larger Δθ = faster generation, coarser pattern

**Typical range:** `5° – 30°`

---

### θ max (π)
**Total angular length of the spiral.**

Measured in multiples of π.

Examples:

- `2π` = one full turn
- `4π` = two turns
- `6π` = three turns

**Effect on robot:**

- Higher values increase total length
- More repeated units
- Larger unfolded sheet size

---

### p
**Internal center spiral ratio / shape factor.**

Controls the relation between outer spiral and center spiral.

**Effect on robot:**

- Changes wall thickness
- Changes internal cavity width
- Modifies trapezoid proportions

**Typical range:** `0.30 – 0.70`

---

## Elastic Layer Parameters

### Elastic %
**Elastic opening factor.**

Controls the auxiliary elastic layer geometry.

**Low values:**

- Stiffer response
- Narrow opening

**High values:**

- More compliance
- Larger opening angle
- More flexible motion

**Typical range:** `5% – 20%`

---

## Derived Values (Displayed in UI)

### Taper Angle
Angle of body narrowing/widening across the full design.

---

### Tip Size
Width of the first segment.

---

### Base Size
Width of the final segment.

---

### Robot Length
Total unfolded body length.

---

# Recommended Starting Presets

## Compact Prototype

a = 1.0  
b = 0.12  
Δθ = 15  
θ max = 4  
p = 0.50  
Elastic = 5%

---

## Flexible Soft Robot

a = 2.0  
b = 0.18  
Δθ = 10  
θ max = 6  
p = 0.55  
Elastic = 12%

---

## Long Tapered Robot

a = 1.2  
b = 0.08  
Δθ = 8  
θ max = 8  
p = 0.50  
Elastic = 8%

---

# Practical Tuning Tips

## If the tip is too large:
Decrease `a`

## If expansion is too aggressive:
Decrease `b`

## If geometry looks coarse:
Decrease `Δθ`

## If robot is too short:
Increase `θ max`

## If walls look too thin/thick:
Adjust `p`

## If motion is too stiff:
Increase `Elastic %`

---

# Fabrication Parameters

## Elastic Layers

**UI Control:** Checkbox  

Enable or disable the generation of elastic intermediate layers between structural units.

When enabled, flexible regions are created between mirrored rigid segments to allow compliant bending.

### Effects

- Adds deformable zones between rigid units  
- Improves flexibility  
- Useful for soft robotic prototypes  

### Recommended Use

- **Enabled:** soft robots / compliant structures  
- **Disabled:** rigid segmented bodies  

---

## Elastic %

**UI Control:** Numeric Spinbox / Slider  

Defines the width of the elastic layer as a percentage of the available taper angle.

### Internal Relation

`elastic_angle = (Elastic% / 100) × (TaperAngle / 2)`

### Interpretation

- **0%** → no elastic opening  
- **25%** → small flexible hinge  
- **50%** → balanced flexibility  
- **100%** → maximum elastic width  

### Recommended Range

- **10–30%** → stiff behavior  
- **30–60%** → balanced  
- **60–100%** → highly compliant  

---

# Tendon Routing Parameters

## Tip Hole Position (%)

**UI Control:** Numeric Spinbox  

Defines radial position of tendon routing hole at the tip.

### Interpretation

- **0%** = center axis  
- **100%** = outer wall  

### Effect

Higher values increase tendon leverage and bending torque.

---

## Tip Hole Size

**UI Control:** Numeric Spinbox  

Diameter of tendon routing hole at the tip section.

### Effect

- Larger holes reduce friction  
- Smaller holes preserve strength  

---

## Base Hole Position (%)

**UI Control:** Numeric Spinbox  

Defines radial position of tendon routing hole at the base.

Used together with Tip Hole Position to define cable path angle.

---

## Base Hole Size

**UI Control:** Numeric Spinbox  

Diameter of tendon routing hole at the base section.

---

# Num of cables (DOF Configuration Mode)
**UI Control:** Checkbox  
## Two Cable Mode (2DOF)

## Three Cable Mode (3DOF)

---

# 2D Structural Parameters

## Extrusion

**UI Control:** Numeric Spinbox / Slider  

Defines thickness used when extruding planar geometry into 3D solids.

### Used In

- 2-cable mode  
- CAD export  
- STL preview  

### Effects

- Higher thickness = stronger structure  
- Lower thickness = lighter / more flexible  

---

## Cone Angle 1

**UI Control:** Numeric Spinbox / Slider  

Defines the tapering angle of the extrusion along the longitudinal **X axis**.

This parameter controls how the total extrusion height progressively decreases in the **Z direction**, generating a pointed or wedge-shaped body.

Instead of trimming only the tip, this parameter applies a continuous sloped reduction of thickness from base to distal end.

### Geometric Meaning

- Higher values create a steeper reduction in height.
- Lower values preserve a more uniform thickness.
- `0°` produces constant extrusion thickness.

### Effect on Robot Shape

- Produces a sharpened distal profile 
- Reduces material near the tip  
- Can improve flexibility at the distal section  
- Alters tendon routing geometry and bending response

### Interpretation

If extrusion thickness is `H` and robot length is `L`, the taper follows:

`Cone1 ≈ 2 · atan((H/2) / L)`

(Automatically constrained by geometry limits)

### Recommended Use

- **Low angle (0–5°):** nearly constant thickness  
- **Medium angle (5–15°):** balanced taper  
- **High angle (>15°):** aggressive pointed geometry


---
## Cone Angle 2

**UI Control:** Numeric Spinbox / Slider  

Defines the tapering intensity in the transverse **Y direction**, modifying the lateral cross-sectional profile of the robot body.

This parameter removes material from the side regions through angled cuts, reshaping the extrusion profile.

### Geometric Meaning

Cone Angle 2 controls how strongly the side corners are trimmed.

- **Low angle:** little lateral trimming  
- **High angle:** aggressive corner removal  

### Effect on Cross Section

#### Low Cone Angle 2

Produces a more squared or rectangular profile.

- Wider lateral edges  
- Higher section stiffness  
- Greater contact surface  

#### High Cone Angle 2

Produces a diamond-like or rhomboid profile.

- Reduced side corners  
- More streamlined geometry  
- Lower transverse stiffness  
- More anisotropic bending response  

### Visual Interpretation

```text
Low Angle            High Angle

 ______              /\ 
|      |            /  \
|______|            \  /
                     \/

```

## Note: 

Cone Angle 2 mainly affects the body shape in YZ cross-section, while Cone Angle 1 controls taper along XZ longitudinal profile.

---

# 3D Fabrication Parameters

(Only used when **3 Cable Mode** is enabled)

---

## Cable3 Cut Enabled

**UI Control:** Checkbox  

Enable manufacturing cuts required for 3-cable radial tendon routing.

These cuts help tendon insertion and improve printability.

---

## Cable3 Cut Position (%)

**UI Control:** Numeric Spinbox  

Defines placement of the 3-cable auxiliary cut.

### Lower Values

Closer to centerline.

### Higher Values

Closer to outer wall.

---

## Cable3 Cut Size (%)

**UI Control:** Numeric Spinbox  

Defines scale of the conical relief cut.

### Effects

- Larger values increase cable clearance  
- Excessive values reduce strength  

---

## Simulation Parameters

These values control joint behavior in MuJoCo simulation.  
They do **not** modify geometry.

---

### Sim Stiffness

Controls joint rotational stiffness (resistance to bending).

- Low → softer structure, bends easily  
- Medium → balanced flexibility  
- High → more rigid, resists motion  

Use higher values if the robot bends too much or feels unstable.

---

### Sim Damping

Controls joint motion damping (resistance while moving).

- Low → faster motion, may oscillate  
- Medium → smoother response  
- High → slower, more stable movement  

Use higher values if the robot vibrates, rebounds, or overshoots.

---


# Practical Recommendations

## Flexible Soft Robot

- Elastic Layers = ON  
- Elastic % = 40%  
- Extrusion = Medium  
- Cone1 = Low  

---

## Rigid Precision Robot

- Elastic Layers = OFF  
- Extrusion = High  
- Cone1 = Small  
- Hole sizes = Tight fit  

---

## High Curvature Tendon Robot

- Tip Hole Position = 80%  
- Base Hole Position = 80%  
- Elastic Layers = ON  
- Cone2 = Medium  

---

# Notes

- Excessive hole size may weaken structure.  
- Large elastic percentages may reduce precision.  
- Cone trimming angles are constrained automatically by geometry.  
- Some parameters are only active depending on cable mode.
"""
§3.3.2  Multi-granularity attribute definition for gas-liquid two-phase flow.

Each of the six canonical flow regimes is described by 12 expert-defined
attributes.  Every attribute is encoded at three levels:
    0   – absent / low
    0.5 – moderate
    1   – strong / definite

Flow regimes implemented
------------------------
0  bubble_flow      泡状流
1  slug_flow        段塞流
2  churn_flow       搅混流
3  annular_flow     环状流
4  stratified_flow  分层流
5  wavy_flow        波状流
"""

import numpy as np

# ── Attribute names (12 attributes) ──────────────────────────────────────────
ATTRIBUTE_NAMES = [
    "bubble_fraction",        # A1  气泡体积分数
    "bubble_size_uniformity", # A2  气泡尺寸均匀度
    "slug_frequency",         # A3  段塞发生频率
    "liquid_film_continuity", # A4  液膜连续性
    "gas_core_continuity",    # A5  气芯连续性
    "interface_waviness",     # A6  界面波动强度
    "flow_periodicity",       # A7  流动周期性
    "pressure_fluctuation",   # A8  压力波动幅度
    "void_fraction",          # A9  截面含气率
    "flow_velocity_ratio",    # A10 气液速度比
    "turbulence_intensity",   # A11 湍流强度
    "stratification_degree",  # A12 分层程度
]

NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)  # 12

# ── Flow-regime labels ────────────────────────────────────────────────────────
FLOW_REGIME_NAMES = [
    "bubble_flow",
    "slug_flow",
    "churn_flow",
    "annular_flow",
    "stratified_flow",
    "wavy_flow",
]

NUM_CLASSES = len(FLOW_REGIME_NAMES)

# ── Attribute matrix  A  (NUM_CLASSES × NUM_ATTRIBUTES) ──────────────────────
# Rows = flow regimes, Columns = attributes
# Values ∈ {0, 0.5, 1}
#
#          A1   A2   A3   A4   A5   A6   A7   A8   A9   A10  A11  A12
ATTRIBUTE_MATRIX = np.array([
    # bubble_flow
    [1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    # slug_flow
    [0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.0],
    # churn_flow
    [0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0],
    # annular_flow
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
    # stratified_flow
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0],
    # wavy_flow
    [0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
], dtype=np.float32)


def get_attribute_vector(class_idx: int) -> np.ndarray:
    """Return the 12-dim attribute vector for a flow-regime class index."""
    return ATTRIBUTE_MATRIX[class_idx].copy()


def get_attribute_matrix() -> np.ndarray:
    """Return a copy of the full attribute matrix (NUM_CLASSES × NUM_ATTRIBUTES)."""
    return ATTRIBUTE_MATRIX.copy()

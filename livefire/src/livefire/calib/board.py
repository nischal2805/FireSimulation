"""ChArUco board definition, print-ready image, detection, and the P1R2 world frame.

Board frame (OpenCV ChArUco): origin at the board's top-left outer corner, +X right along the
``squares_x`` direction, +Y down, +Z into the page. World frame (P1R2): origin at the board centre,
+X = board X, +Z = board normal pointing toward the viewer (cameras), +Y = Z x X = -board Y; the
printed face lies in z = 0. Units metres, host NumPy float64, caller owns the returned arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.common.config import ConfigError, Params, load_yaml

A4_LANDSCAPE_MM = (297.0, 210.0)


class CalibrationError(ValueError):
    """Calibration missing, malformed, or inconsistent with the frames it is applied to."""


@dataclass(frozen=True)
class BoardSpec:
    """Printed ChArUco board layout (metres)."""

    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.035
    marker_length_m: float = 0.026
    dictionary: str = "DICT_5X5_100"

    def __post_init__(self) -> None:
        if not (0.0 < self.marker_length_m < self.square_length_m):
            raise ValueError("need 0 < marker_length_m < square_length_m")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown aruco dictionary {self.dictionary}")

    @property
    def size_m(self) -> tuple[float, float]:
        """(width, height) of the board in metres."""
        return self.squares_x * self.square_length_m, self.squares_y * self.square_length_m

    def to_dict(self) -> dict:
        return {"squares_x": self.squares_x, "squares_y": self.squares_y, "square_length_m": self.square_length_m,
                "marker_length_m": self.marker_length_m, "dictionary": self.dictionary}

    @staticmethod
    def from_dict(d: dict) -> BoardSpec:
        return BoardSpec(int(d["squares_x"]), int(d["squares_y"]), float(d["square_length_m"]),
                         float(d["marker_length_m"]), str(d["dictionary"]))


def load_board_spec(cfg: dict | None = None) -> BoardSpec:
    """BoardSpec from ``configs/calibration.yaml`` (or a given parsed mapping)."""
    cfg = load_yaml("calibration") if cfg is None else cfg
    try:
        b = cfg["board"]
        p = Params({k: b[k] for k in ("square_length_m", "marker_to_square_ratio")})
        sq = p["square_length_m"]
        return BoardSpec(int(b["squares_x"]), int(b["squares_y"]), sq, sq * p["marker_to_square_ratio"],
                         str(b["dictionary"]))
    except KeyError as e:
        raise ConfigError(f"calibration.yaml board: missing {e}") from None


def make_board(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    """OpenCV 5 CharucoBoard for ``spec``."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    return cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_length_m, spec.marker_length_m, d)


def board_corners_board(spec: BoardSpec) -> NDArray[np.float64]:
    """All inner ChArUco corners in the board frame, (M,3) m, indexed by corner id."""
    return np.asarray(make_board(spec).getChessboardCorners(), dtype=np.float64).reshape(-1, 3)


def T_world_from_board(spec: BoardSpec) -> NDArray[np.float64]:
    """(4,4) board frame -> world frame (P1R2): origin at the board centre, +Z toward the viewer."""
    w, h = spec.size_m
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = -T[:3, :3] @ np.array([w / 2.0, h / 2.0, 0.0])
    return T


def board_corners_world(spec: BoardSpec) -> NDArray[np.float64]:
    """Inner ChArUco corners in the world frame, (M,3) m (z = 0)."""
    p = board_corners_board(spec)
    T = T_world_from_board(spec)
    return p @ T[:3, :3].T + T[:3, 3]


def render_board_png(spec: BoardSpec, dpi: int = 300, page_mm: tuple[float, float] | None = A4_LANDSCAPE_MM):
    """Print-ready uint8 (H,W) image of the board centred on a white page (print at 100 % scale).

    Returns ``(image, info)``; ``info['board_mm']`` is the physical board size, ``info['page_mm']`` the page.
    """
    px_per_m = dpi / 0.0254
    bw, bh = spec.size_m
    bw_px, bh_px = round(bw * px_per_m), round(bh * px_per_m)
    board_img = make_board(spec).generateImage((bw_px, bh_px), marginSize=0, borderBits=1)
    if page_mm is None:
        page = board_img
        page_mm = (bw * 1000.0, bh * 1000.0)
    else:
        pw, ph = round(page_mm[0] / 25.4 * dpi), round(page_mm[1] / 25.4 * dpi)
        if pw < bw_px or ph < bh_px:
            raise ValueError("board does not fit on the page at this scale")
        page = np.full((ph, pw), 255, np.uint8)
        y0, x0 = (ph - bh_px) // 2, (pw - bw_px) // 2
        page[y0 : y0 + bh_px, x0 : x0 + bw_px] = board_img
    return page, {"board_mm": (bw * 1000.0, bh * 1000.0), "page_mm": page_mm, "dpi": dpi,
                  "square_mm": spec.square_length_m * 1000.0, "marker_mm": spec.marker_length_m * 1000.0}


def detect_charuco(spec: BoardSpec, image: NDArray, *, detector: cv2.aruco.CharucoDetector | None = None):
    """Detect ChArUco corners. ``image`` uint8 (H,W) or (H,W,3) BGR, distorted pixels.

    Returns ``(corners (N,2) float64 px, ids (N,) int)``; N = 0 when nothing is found."""
    det = detector or cv2.aruco.CharucoDetector(make_board(spec))
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c, ids, _, _ = det.detectBoard(gray)
    if c is None or ids is None or len(ids) == 0:
        return np.zeros((0, 2)), np.zeros((0,), np.int32)
    return np.asarray(c, np.float64).reshape(-1, 2), np.asarray(ids, np.int32).reshape(-1)


def object_points(spec: BoardSpec, ids: NDArray[np.int32]) -> NDArray[np.float64]:
    """Board-frame 3D points (N,3) m for corner ids."""
    return board_corners_board(spec)[np.asarray(ids, dtype=int)]


def solve_board_pose(
    spec: BoardSpec, corners: NDArray, ids: NDArray, K: NDArray, dist: NDArray
) -> tuple[NDArray[np.float64], float]:
    """PnP of the detected corners (distorted px, camera ``K``/``dist``).

    Returns ``(T_camera_from_board (4,4), rms_px)`` where rms is over the corners used.
    Planar IPPE start, then Levenberg-Marquardt refinement. Needs >= 4 corners."""
    obj = object_points(spec, ids)
    img = np.ascontiguousarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    if len(obj) < 4:
        raise ValueError(f"need >= 4 corners, got {len(obj)}")
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    return T, float(np.sqrt(np.mean(err**2)))

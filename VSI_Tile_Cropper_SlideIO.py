from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cv2
import numpy as np

# --- optional writers ---
try:
    import imageio.v3 as imageio  # better PNG writer (can handle uint16 RGB if imagecodecs is present)
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    from PIL import Image, PngImagePlugin
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# --- SlideIO (required) ---
try:
    import slideio
except ImportError as e:
    raise SystemExit("Please install SlideIO: `pip install -U slideio`") from e


# -----------------------
# Types & Configuration
# -----------------------
Rect = Tuple[int, int, int, int]
Size = Tuple[int, int]


@dataclass
class ProjectionConfig:
    """
    Tunable settings for the Pseudo-RGB Z-projection.
    Channel mapping:
      - Red   (0): CLAHE(Best-focus single Z-plane via Laplacian variance)
      - Green (1): CLAHE(Min intensity projection across Z)
      - Blue  (2): CLAHE(Max intensity projection across Z)
    """
    clahe_clip: float = 2.5
    clahe_grid: Tuple[int, int] = (16, 16)
    focus_downsample_step: int = 2  # Subsample step for fast Laplacian focus scoring (1=full res, 2=2x faster)


def _ensure_tuple2(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


def _scene_area(scene) -> int:
    x, y, w, h = scene.rect  # documented property (x, y, width, height)
    return int(w) * int(h)


def _pick_default_scene(slide) -> int:
    areas = [(_scene_area(slide.get_scene(i)), i) for i in range(slide.num_scenes)]
    areas.sort(reverse=True)
    return areas[0][1] if areas else 0


def _grid(rect: Rect, tile: Size, overlap: Size) -> Iterable[Rect]:
    """
    Generate tile rectangles that cover 'rect' with the given tile size and overlap.
    rect: (x0, y0, W, H)  [scene coordinates, pixels]
    """
    x0, y0, W, H = rect
    tw, th = tile
    ox, oy = overlap

    if tw <= 0 or th <= 0:
        raise ValueError("tile size must be positive")
    if ox >= tw or oy >= th:
        raise ValueError("overlap must be smaller than tile size")

    step_x = tw - ox
    step_y = th - oy

    y = y0
    while y < y0 + H:
        x = x0
        tile_h = min(th, y0 + H - y)
        while x < x0 + W:
            tile_w = min(tw, x0 + W - x)
            yield (x, y, tile_w, tile_h)
            x += step_x
        y += step_y


# -----------------------
# Z-Stack & Projection Helpers
# -----------------------
def _extract_z_slices_raw(block: np.ndarray, num_z: int) -> List[np.ndarray]:
    """
    Normalize SlideIO `read_block(..., slices=(0, num_z))` output into a list of
    `num_z` 2D/3D single-plane arrays (H, W) or (H, W, C).
    """
    if num_z == 1:
        if block.ndim == 4 and block.shape[-1] == 1:
            return [np.squeeze(block, axis=-1)]
        return [block]

    # SlideIO returns (H, W, Z) for 1-channel or (H, W, C, Z) for multi-channel
    if block.ndim == 3 and block.shape[-1] == num_z:
        return [block[:, :, z] for z in range(num_z)]
    if block.ndim == 4 and block.shape[-1] == num_z:
        return [block[:, :, :, z] for z in range(num_z)]

    # Fallback if Z is axis 0
    return [block[z] for z in range(num_z)]


def _to_grayscale_u8(plane: np.ndarray) -> np.ndarray:
    """Convert a single Z-plane (H, W) or (H, W, C) to uint8 grayscale for projection."""
    if plane.dtype == np.uint16:
        arr = (plane >> 8).astype(np.uint8)
    elif plane.dtype != np.uint8:
        arr = np.clip(plane, 0, 255).astype(np.uint8)
    else:
        arr = plane

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[:, :, 0]
        if arr.shape[-1] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _build_pseudo_rgb(
    gray_planes: Sequence[np.ndarray],
    cfg: ProjectionConfig,
) -> np.ndarray:
    """
    Compute a 3-channel Pseudo-RGB uint8 image (H, W, 3) from a sequence of
    uint8 grayscale Z-planes without allocating a 3D stack in memory.

    Channels:
      - Red   (0): CLAHE(Best single Z-plane, chosen via Laplacian variance)
      - Green (1): CLAHE(Min intensity projection across Z)
      - Blue  (2): CLAHE(Max intensity projection across Z)
    """
    p_min: Optional[np.ndarray] = None
    p_max: Optional[np.ndarray] = None
    best_slice: Optional[np.ndarray] = None
    best_focus_score: float = -1.0

    step = max(1, int(cfg.focus_downsample_step))

    for gray in gray_planes:
        # 1. Update running Min and Max intensity projections in-place
        if p_min is None:
            p_min = gray.copy()
            p_max = gray.copy()
        else:
            np.minimum(p_min, gray, out=p_min)
            np.maximum(p_max, gray, out=p_max)

        # 2. Score focal sharpness using Laplacian variance on strided grid
        sub = gray[::step, ::step] if step > 1 else gray
        focus_score = float(cv2.Laplacian(sub, cv2.CV_16S, ksize=3).var())
        if focus_score > best_focus_score:
            best_focus_score = focus_score
            best_slice = gray

    # 3. Apply local contrast enhancement (CLAHE) independently per channel
    # (Created per-call to remain 100% thread-safe across ThreadPoolExecutor workers)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)

    ch_red_best_z = clahe.apply(best_slice)
    ch_green_min  = clahe.apply(p_min)
    ch_blue_max   = clahe.apply(p_max)

    return np.stack([ch_red_best_z, ch_green_min, ch_blue_max], axis=-1)


# -----------------------
# PNG writer
# -----------------------
def _write_png(
    arr: np.ndarray,
    path: Union[str, Path],
    compress_level: int = 2,  # lossless; 0=store, 9=max compression
) -> None:
    """
    Lossless PNG writer with best-effort dtype preservation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_IMAGEIO:
        imageio.imwrite(path, arr, extension=".png")
        return

    if not _HAS_PIL:
        np.save(path.with_suffix(".npy"), arr)
        return

    out = np.ascontiguousarray(arr)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Software", "vsi_tiler (SlideIO)")

    if out.ndim == 2:
        if out.dtype == np.uint16:
            im = Image.fromarray(out, mode="I;16")
        else:
            im = Image.fromarray(out)
        im.save(path, format="PNG", pnginfo=pnginfo, compress_level=compress_level)
        return

    if out.ndim == 3 and out.shape[-1] in (3, 4):
        if out.dtype == np.uint16:
            out8 = (out >> 8).astype(np.uint8)
            im = Image.fromarray(out8)
        else:
            im = Image.fromarray(out)
        im.save(path, format="PNG", pnginfo=pnginfo, compress_level=compress_level)
        return

    np.save(path.with_suffix(".npy"), out)


# -----------------------
# Metadata (JSON)
# -----------------------
@dataclass
class OMEDetails:
    calibration: Optional[float]  # micrometers per pixel (X)
    calibration_unit: str
    width: int
    height: int
    z_plane: int  # highest z index


@dataclass
class RunMetadata:
    tile_size: int
    tile_overlap: int
    ome_metadata: OMEDetails
    total_tile_count: int
    processed_tile_count: int
    percent_complete: float

    slide_name: str = ""
    scene_index: int = 0
    num_z_slices: int = 1
    channels: Optional[List[int]] = None
    tile_naming: str = "{x}x_{y}y.png"
    saved_individual_z_planes: bool = False

    def to_json(self) -> Dict:
        base = {
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "ome_metadata": {
                "calibration": self.ome_metadata.calibration,
                "calibration_unit": self.ome_metadata.calibration_unit,
                "width": self.ome_metadata.width,
                "height": self.ome_metadata.height,
                "z_plane": self.ome_metadata.z_plane,
            },
            "total_tile_count": self.total_tile_count,
            "processed_tile_count": self.processed_tile_count,
            "percent_complete": self.percent_complete,
        }
        base.update({
            "slide_name": self.slide_name,
            "scene_index": self.scene_index,
            "num_z_slices": self.num_z_slices,
            "channels": self.channels,
            "tile_naming": self.tile_naming,
            "saved_individual_z_planes": self.saved_individual_z_planes,
        })
        return base


# -----------------------
# Main tiler
# -----------------------
def crop_vsi_tiles(
    vsi_path: Union[str, Path],
    out_root: Union[str, Path],
    scene_index: Optional[int] = None,
    tile_size: Union[int, Size] = 2048,
    overlap: Union[int, Size] = 512,
    channels: Optional[Sequence[int]] = None,
    num_workers: Optional[int] = None,
    png_compress_level: int = 2,
    save_z_planes: bool = False,
    projection_cfg: Optional[ProjectionConfig] = None,
) -> Dict:
    """
    Tile an Olympus .vsi (including z-stacks) into single-plane pseudo-RGB PNG tiles
    saved directly as `<out_root>/<slide_name>/<rx>x_<ry>y.png`
    (R=CLAHE(Best-Z), G=CLAHE(Min), B=CLAHE(Max)).

    If `save_z_planes=True`, also saves individual focal planes into `<rx>x_<ry>y/<z>z.png`.
    """
    proj_cfg = projection_cfg or ProjectionConfig()
    vsi_path = Path(vsi_path)
    slide_name = vsi_path.stem

    # Open VSI
    slide = slideio.open_slide(str(vsi_path), "VSI")
    if scene_index is None:
        scene_index = _pick_default_scene(slide)
    scene = slide.get_scene(scene_index)

    # Scene geometry & Z
    x0, y0, W, H = scene.rect
    num_z = scene.num_z_slices

    # Pixel size (μm/px)
    px_size_um = None
    try:
        rx_m, ry_m = scene.get_resolution()
        px_size_um = float(rx_m) * 1e6
    except Exception:
        try:
            lvl0 = scene.get_zoom_level_info(0)
            rx_m = getattr(lvl0, "resolution_x", None)
            if rx_m is not None:
                px_size_um = float(rx_m) * 1e6
        except Exception:
            px_size_um = None

    # Prepare output directories
    out_root = Path(out_root)
    parent_dir = out_root / slide_name
    parent_dir.mkdir(parents=True, exist_ok=True)
    meta_path = parent_dir / "metadata.json"

    # Grid
    tile_size = _ensure_tuple2(tile_size)
    overlap = _ensure_tuple2(overlap)
    rects = list(_grid((x0, y0, W, H), tile_size, overlap))
    xy_tile_count = len(rects)

    # Channels
    channel_indices = [] if channels is None else list(channels)

    # Seed metadata JSON
    md = RunMetadata(
        tile_size=int(tile_size[0]),
        tile_overlap=int(overlap[0]),
        ome_metadata=OMEDetails(
            calibration=px_size_um,
            calibration_unit="µm",
            width=int(W),
            height=int(H),
            z_plane=int(num_z - 1),
        ),
        total_tile_count=xy_tile_count,
        processed_tile_count=0,
        percent_complete=0.0,
        slide_name=slide_name,
        scene_index=int(scene_index),
        num_z_slices=int(num_z),
        channels=channel_indices or None,
        tile_naming="{x}x_{y}y.png",
        saved_individual_z_planes=bool(save_z_planes),
    )

    def _flush_meta():
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(md.to_json(), f, indent=4)

    _flush_meta()

    # Worker that reads all Z-slices in a single call and writes `<parent_dir>/<rx>x_<ry>y.png`
    def _process_xy_tile(r: Rect):
        rx, ry, rw, rh = r
        tile_stem = f"{rx}x_{ry}y"

        # Single SlideIO read for the entire Z-stack
        block = scene.read_block(
            rect=(rx, ry, rw, rh),
            size=(0, 0),
            channel_indices=channel_indices,
            slices=(0, num_z),
        )
        raw_planes = _extract_z_slices_raw(block, num_z)

        # Optional: write individual Z-planes into a subfolder only if explicitly requested
        if save_z_planes:
            z_dir = parent_dir / tile_stem
            z_dir.mkdir(parents=True, exist_ok=True)
            for z, raw_plane in enumerate(raw_planes):
                _write_png(raw_plane, z_dir / f"{z}z.png", compress_level=png_compress_level)

        # Build and write the Pseudo-RGB projection directly as `<parent_dir>/<rx>x_<ry>y.png`
        gray_planes = [_to_grayscale_u8(p) for p in raw_planes]
        pseudo_rgb = _build_pseudo_rgb(gray_planes, cfg=proj_cfg)
        out_path = parent_dir / f"{tile_stem}.png"
        _write_png(pseudo_rgb, out_path, compress_level=png_compress_level)

        return (rx, ry)

    # Threading: one job per (x,y) tile
    max_workers = num_workers or min(8, os.cpu_count() or 4)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_xy_tile, r) for r in rects]

        pbar = tqdm(total=xy_tile_count, desc=f"{slide_name} tiles", unit="tile")
        try:
            for fut in as_completed(futures):
                _ = fut.result()
                completed += 1
                md.processed_tile_count = completed
                md.percent_complete = round(100.0 * completed / xy_tile_count, 3)
                # Throttle metadata disk writes to every 25 tiles to avoid I/O bottlenecks
                if completed % 25 == 0:
                    _flush_meta()
                pbar.update(1)
        finally:
            pbar.close()

    _flush_meta()
    return md.to_json()


# -----------------------
# Batch helpers
# -----------------------
def discover_vsi_files(
    batch_dir: Union[str, Path],
    pattern: str = "*.vsi",
    recursive: bool = True,
) -> List[Path]:
    """
    Return sorted list of .vsi files in a directory (optionally recursive)
    using an internal glob pattern (default: '*.vsi').
    """
    base = Path(batch_dir)
    if not base.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    if recursive:
        files = sorted(base.rglob(pattern))
    else:
        files = sorted(base.glob(pattern))
    return [p for p in files if p.is_file()]


def batch_crop_vsi_tiles(
    inputs: Sequence[Union[str, Path]],
    out_root: Union[str, Path],
    *,
    scene_index: Optional[int] = None,
    tile_size: Union[int, Size] = 2048,
    overlap: Union[int, Size] = 512,
    channels: Optional[Sequence[int]] = None,
    num_workers: Optional[int] = None,
    png_compress_level: int = 2,
    save_z_planes: bool = False,
    projection_cfg: Optional[ProjectionConfig] = None,
    slides_workers: int = 1,
    skip_existing: bool = False,
) -> List[Dict]:
    """
    Process multiple slides. Each slide's tiles go to:
        <out_root>/<VSI_STEM>/<rx>x_<ry>y.png
    Returns a list of result dicts (one per slide).
    """
    uniq: List[Path] = []
    seen = set()
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for f in sorted(p.glob("*.vsi")):
                if f not in seen:
                    uniq.append(f); seen.add(f)
        else:
            if p not in seen:
                uniq.append(p); seen.add(p)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    def _one(vsi_path: Path) -> Dict:
        slide_name = vsi_path.stem
        slide_out_dir = out_root / slide_name
        meta_file = slide_out_dir / "metadata.json"

        if skip_existing and meta_file.exists():
            return {"slide": slide_name, "status": "skipped", "path": str(slide_out_dir)}

        md = crop_vsi_tiles(
            vsi_path=vsi_path,
            out_root=out_root,
            scene_index=scene_index,
            tile_size=tile_size,
            overlap=overlap,
            channels=channels,
            num_workers=num_workers,
            png_compress_level=png_compress_level,
            save_z_planes=save_z_planes,
            projection_cfg=projection_cfg,
        )
        return {"slide": slide_name, "status": "ok", "path": str(slide_out_dir), "meta": md}

    if slides_workers and slides_workers > 1:
        with ThreadPoolExecutor(max_workers=slides_workers) as ex:
            futs = [ex.submit(_one, p) for p in uniq]
            pbar = tqdm(total=len(futs), desc="Slides", unit="slide")
            try:
                for fut in as_completed(futs):
                    results.append(fut.result())
                    pbar.update(1)
            finally:
                pbar.close()
    else:
        for p in tqdm(uniq, desc="Slides", unit="slide"):
            results.append(_one(p))

    return results


# -----------------------
# CLI (single or batch)
# -----------------------
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Tile Olympus VSI (z-stacks) into single-plane pseudo-RGB PNG tiles (<x>x_<y>y.png)."
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vsi", type=str, help="Path to a single .vsi file.")
    src.add_argument("--batch-dir", type=str, help="Process all *.vsi files in this directory.")
    src.add_argument("--list", type=str, help="Text file with one .vsi path per line.")

    ap.add_argument("out", type=str, help="Output root directory.")

    # Tiling / reading options
    ap.add_argument("--scene", type=int, default=None, help="Scene index (default: largest scene).")
    ap.add_argument("--tile", type=int, nargs=2, default=(2048, 2048), metavar=("W", "H"))
    ap.add_argument("--overlap", type=int, nargs=2, default=(512, 512), metavar=("OX", "OY"))
    ap.add_argument("--channels", type=int, nargs="*", default=None, help="Channel indices (default: all).")
    ap.add_argument("--workers", type=int, default=None, help="Per-slide tile thread count (default: up to 8).")
    ap.add_argument("--png-compress", type=int, default=2, choices=range(0, 10), help="PNG compression (0..9, default: 2).")

    # Projection & Z-plane options
    ap.add_argument("--save-z-planes", action="store_true", help="Also write individual <z>z.png slices into <x>x_<y>y/ subfolders (disabled by default).")
    ap.add_argument("--clahe-clip", type=float, default=2.5, help="CLAHE clip limit (default: 2.5).")
    ap.add_argument("--clahe-grid", type=int, nargs=2, default=(16, 16), metavar=("GX", "GY"), help="CLAHE tile grid size (default: 16 16).")
    ap.add_argument("--focus-step", type=int, default=2, help="Downsample stride for Laplacian focus scoring (default: 2).")

    # Batch options
    ap.add_argument("--pattern", type=str, default="*.vsi", help="Glob pattern for --batch-dir (default: *.vsi).")
    ap.add_argument("--recursive", action="store_true", help="Recurse subfolders for --batch-dir.")
    ap.add_argument("--slides-workers", type=int, default=1, help="Parallel slides count (default: 1).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip slides that already have metadata.json.")
    ap.add_argument("--dry-run", action="store_true", help="List slides that would be processed and exit.")

    args = ap.parse_args()

    proj_cfg = ProjectionConfig(
        clahe_clip=args.clahe_clip,
        clahe_grid=tuple(args.clahe_grid),
        focus_downsample_step=args.focus_step,
    )

    if args.vsi:
        md = crop_vsi_tiles(
            vsi_path=args.vsi,
            out_root=args.out,
            scene_index=args.scene,
            tile_size=tuple(args.tile),
            overlap=tuple(args.overlap),
            channels=args.channels,
            num_workers=args.workers,
            png_compress_level=args.png_compress,
            save_z_planes=args.save_z_planes,
            projection_cfg=proj_cfg,
        )
        print(json.dumps(md, indent=2))
        sys.exit(0)

    slides: List[Path] = []
    if args.batch_dir:
        slides.extend(discover_vsi_files(args.batch_dir, pattern=args.pattern, recursive=args.recursive))
    if args.list:
        list_file = Path(args.list)
        if not list_file.exists():
            ap.error(f"--list file not found: {args.list}")
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    slides.append(Path(line))

    seen = set()
    slides = [p for p in slides if (p not in seen and not seen.add(p))]

    if args.dry_run:
        print(f"[DRY-RUN] Found {len(slides)} slides:")
        for p in slides:
            print(" -", p)
        sys.exit(0)

    if not slides:
        ap.error("No .vsi slides found for batch processing.")

    results = batch_crop_vsi_tiles(
        inputs=slides,
        out_root=args.out,
        scene_index=args.scene,
        tile_size=tuple(args.tile),
        overlap=tuple(args.overlap),
        channels=args.channels,
        num_workers=args.workers,
        png_compress_level=args.png_compress,
        save_z_planes=args.save_z_planes,
        projection_cfg=proj_cfg,
        slides_workers=args.slides_workers,
        skip_existing=args.skip_existing,
    )

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"[DONE] Slides processed: {ok}, skipped: {skipped}, total: {len(results)}")
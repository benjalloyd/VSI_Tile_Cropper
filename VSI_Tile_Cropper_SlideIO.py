from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
# Types & helpers
# -----------------------
Rect = Tuple[int, int, int, int]
Size = Tuple[int, int]


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
# PNG writer
# -----------------------
def _write_png(
    arr: np.ndarray,
    path: Union[str, Path],
    compress_level: int = 4,  # lossless; 0=store, 9=max compression
) -> None:
    """
    Lossless PNG writer with best-effort dtype preservation.
    - If imageio is available (plus imagecodecs), this preserves uint16 RGB/Gray.
    - Else Pillow: preserves uint8 RGB/Gray; preserves uint16 Gray; may downcast uint16 RGB.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_IMAGEIO:
        # imageio chooses best backend; with imagecodecs it writes 16-bit PNG cleanly
        imageio.imwrite(path, arr, extension=".png")  # lossless by definition
        return

    if not _HAS_PIL:
        # last resort – NumPy (so data are not lost)
        np.save(path.with_suffix(".npy"), arr)
        return

    # Pillow backend
    # Ensure writeable contiguous
    out = np.ascontiguousarray(arr)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Software", "vsi_tiler (SlideIO)")

    if out.ndim == 2:
        # Gray
        if out.dtype == np.uint16:
            im = Image.fromarray(out, mode="I;16")
        else:
            im = Image.fromarray(out)
        im.save(path, format="PNG", pnginfo=pnginfo, compress_level=compress_level)
        return

    if out.ndim == 3 and out.shape[-1] in (3, 4):
        # RGB / RGBA
        if out.dtype == np.uint16:
            # Pillow often downcasts 16-bit RGB PNG. To avoid surprises,
            # we right-shift to 8-bit while keeping dynamic range shape.
            out8 = (out >> 8).astype(np.uint8)
            im = Image.fromarray(out8)
        else:
            im = Image.fromarray(out)
        im.save(path, format="PNG", pnginfo=pnginfo, compress_level=compress_level)
        return

    # Otherwise, save NumPy for full fidelity
    np.save(path.with_suffix(".npy"), out)


# -----------------------
# Metadata (JSON)
# -----------------------
@dataclass
class OMEDetails:
    calibration: Optional[float]  # micrometers per pixel (X); Y assumed same if None elsewhere
    calibration_unit: str
    width: int
    height: int
    z_plane: int  # highest z index (for your schema)


@dataclass
class RunMetadata:
    tile_size: int
    tile_overlap: int
    ome_metadata: OMEDetails
    total_tile_count: int
    processed_tile_count: int
    percent_complete: float

    # Optional extras (kept out of your schema, but helpful):
    slide_name: str = ""
    scene_index: int = 0
    num_z_slices: int = 1
    channels: Optional[List[int]] = None

    def to_json(self) -> Dict:
        # Keep your schema 100% intact; extras added at the end
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
        # Add non-breaking extras
        base.update({
            "slide_name": self.slide_name,
            "scene_index": self.scene_index,
            "num_z_slices": self.num_z_slices,
            "channels": self.channels,
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
    png_compress_level: int = 4,  # lossless; 0 fastest/big, 9 slowest/small
) -> Dict:
    """
    Tile an Olympus .vsi (including z-stacks) into PNG tiles.
    Directory structure and JSON exactly as requested.

    Returns the final metadata dictionary.
    """
    vsi_path = Path(vsi_path)
    slide_name = vsi_path.stem

    # Open VSI
    slide = slideio.open_slide(str(vsi_path), "VSI")  # VSI driver [1](https://www.slideio.com/sphinx/tutorial.html)
    if scene_index is None:
        scene_index = _pick_default_scene(slide)
    scene = slide.get_scene(scene_index)

    # Scene geometry & Z
    x0, y0, W, H = scene.rect               # rect = (x, y, width, height) [2](https://www.slideio.com/sphinx/scene.html)
    num_z = scene.num_z_slices              # property num_z_slices           [2](https://www.slideio.com/sphinx/scene.html)

    # Pixel size (μm/px). Try multiple APIs depending on SlideIO version.
    px_size_um = None
    try:
        # Many bindings provide get_resolution() → (x_res_m, y_res_m)
        rx_m, ry_m = scene.get_resolution()  # meters per pixel (x, y)  (C++ API has this) [7](https://www.slideio.com/doxygen/html/classslideio_1_1Scene.html)
        px_size_um = float(rx_m) * 1e6
    except Exception:
        try:
            # Sometimes accessible via pyramid level info
            lvl0 = scene.get_zoom_level_info(0)  # may include resolution downsamples (not strictly documented)
            # Best effort; if missing, leave None
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

    # Seed metadata JSON (your schema)
    md = RunMetadata(
        tile_size=int(tile_size[0]),            # square per your default
        tile_overlap=int(overlap[0]),
        ome_metadata=OMEDetails(
            calibration=px_size_um,
            calibration_unit="µm",
            width=int(W),
            height=int(H),
            z_plane=int(num_z - 1),            # highest z index to mirror your sample
        ),
        total_tile_count=xy_tile_count,         # count of (x,y) tiles, as in your sample
        processed_tile_count=0,
        percent_complete=0.0,
        slide_name=slide_name,
        scene_index=int(scene_index),
        num_z_slices=int(num_z),
        channels=channel_indices or None,
    )

    def _flush_meta():
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(md.to_json(), f, indent=4)

    _flush_meta()

    # Worker that writes ALL z-slices for one rect into "<slide>/<xx_x>/<z>.png"
    def _process_xy_tile(r: Rect):
        rx, ry, rw, rh = r
        tile_dir = parent_dir / f"{rx}x_{ry}y"
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Iterate z slices (0..num_z-1). `slices=(z,z+1)` per SlideIO spec. [3](https://www.slideio.com/sphinx/functions.html)
        for z in range(num_z):
            arr = scene.read_block(
                rect=(rx, ry, rw, rh),
                size=(0, 0),  # no rescaling; use pyramid only if you decide to add it
                channel_indices=channel_indices,
                slices=(z, z + 1),
            )  # Returns numpy array; z dimension size==1 for a single slice. [1](https://www.slideio.com/sphinx/tutorial.html)

            # If reader returns an extra slice axis, squeeze it
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)

            out_path = tile_dir / f"{z}z.png"
            _write_png(arr, out_path, compress_level=png_compress_level)

        # Mark this (x,y) tile as processed
        return (rx, ry)

    # Threading: one job per (x,y) tile
    max_workers = num_workers or (os.cpu_count() or 4)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_xy_tile, r) for r in rects]

        # Nice terminal progress bar that shows tile completion progress
        pbar = tqdm(total=xy_tile_count, desc=f"{slide_name} tiles", unit="tile")
        try:
            for fut in as_completed(futures):
                _ = fut.result()  # propagate any exceptions
                completed += 1
                md.processed_tile_count = completed
                md.percent_complete = round(100.0 * completed / xy_tile_count, 3)
                _flush_meta()
                pbar.update(1)
        finally:
            pbar.close()

    # Final write
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
    png_compress_level: int = 4,
    slides_workers: int = 1,
    skip_existing: bool = False,
) -> List[Dict]:
    """
    Process multiple slides. Each slide's tiles go to:
        <out_root>/<VSI_STEM>/
    Returns a list of result dicts (one per slide).
    """
    # Normalize & de-duplicate inputs
    uniq: List[Path] = []
    seen = set()
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            # If a directory sneaks in here, treat it like "non-recursive" add:
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
            out_root=out_root,               # IMPORTANT: crop_vsi_tiles adds <slide_name> subdir itself
            scene_index=scene_index,
            tile_size=tile_size,
            overlap=overlap,
            channels=channels,
            num_workers=num_workers,
            png_compress_level=png_compress_level,
        )
        return {"slide": slide_name, "status": "ok", "path": str(slide_out_dir), "meta": md}

    # Optional: parallelize across slides (each slide uses its own per-tile threads)
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
        description="Tile Olympus VSI (z-stacks) into lossless PNG tiles."
    )

    # Mutually-exclusive input sources:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--vsi",
        type=str,
        help="Path to a single .vsi file (kept backward compatible with your previous CLI).",
    )
    src.add_argument(
        "--batch-dir",
        type=str,
        help="Process all *.vsi files found in this directory (use --recursive / --pattern for control).",
    )
    src.add_argument(
        "--list",
        type=str,
        help="Text file with one absolute/relative .vsi path per line (blank lines and lines starting with '#' are ignored).",
    )

    ap.add_argument(
        "out",
        type=str,
        help="Output root directory. Each slide will get a subfolder named after the VSI (no extension).",
    )

    # Tiling / reading options (shared)
    ap.add_argument("--scene", type=int, default=None, help="Scene index (default: largest scene).")
    ap.add_argument("--tile", type=int, nargs=2, default=(2048, 2048), metavar=("W", "H"))
    ap.add_argument("--overlap", type=int, nargs=2, default=(512, 512), metavar=("OX", "OY"))
    ap.add_argument("--channels", type=int, nargs="*", default=None, help="Channel indices (default: all)")
    ap.add_argument("--workers", type=int, default=None, help="Per-slide tile thread count (default: CPU count).")
    ap.add_argument("--png-compress", type=int, default=4, choices=range(0, 10), help="PNG compression (0..9), lossless.")

    # Batch discovery / execution options
    ap.add_argument("--pattern", type=str, default="*.vsi", help="Glob pattern when using --batch-dir (default: *.vsi).")
    ap.add_argument("--recursive", action="store_true", help="Recurse subfolders for --batch-dir.")
    ap.add_argument("--slides-workers", type=int, default=1, help="Number of slides to process in parallel (default: 1).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip slides whose output folder already has metadata.json.")
    ap.add_argument("--dry-run", action="store_true", help="List the slides that would be processed and exit.")

    args = ap.parse_args()

    # Build input list for batch mode
    if args.vsi:
        # Single slide (keeps compatibility with your previous CLI)
        md = crop_vsi_tiles(
            vsi_path=args.vsi,
            out_root=args.out,
            scene_index=args.scene,
            tile_size=tuple(args.tile),
            overlap=tuple(args.overlap),
            channels=args.channels,
            num_workers=args.workers,
            png_compress_level=args.png_compress,
        )
        # Print a small summary
        print(json.dumps(md, indent=2))
        sys.exit(0)

    slides: List[Path] = []
    if args.batch_dir:
        slides.extend(
            discover_vsi_files(
                batch_dir=args.batch_dir,
                pattern=args.pattern,
                recursive=args.recursive,
            )
        )
    if args.list:
        list_file = Path(args.list)
        if not list_file.exists():
            ap.error(f"--list file not found: {args.list}")
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                slides.append(Path(line))

    # De-duplicate while preserving order
    seen = set()
    slides = [p for p in slides if (p not in seen and not seen.add(p))]

    if args.dry_run:
        print(f"[DRY-RUN] Found {len(slides)} slides:")
        for p in slides:
            print(" -", p)
        sys.exit(0)

    if not slides:
        ap.error("No .vsi slides found for batch processing.")

    # Run the batch
    results = batch_crop_vsi_tiles(
        inputs=slides,
        out_root=args.out,
        scene_index=args.scene,
        tile_size=tuple(args.tile),
        overlap=tuple(args.overlap),
        channels=args.channels,
        num_workers=args.workers,
        png_compress_level=args.png_compress,
        slides_workers=args.slides_workers,
        skip_existing=args.skip_existing,
    )

    # Print a compact summary
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"[DONE] Slides processed: {ok}, skipped: {skipped}, total: {len(results)}")

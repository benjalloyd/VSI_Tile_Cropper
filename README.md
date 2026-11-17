# VSI_Tile_Cropper
Tool to rapidly crop tiles from Olympus VSI whole-slide images. Capabilities include Z-stack processing, custom size/overlap of cropped tiles.

## Required Packages
 - __ future __  (annotations)
 - json
 - os
 - dataclasses (dataclass)
 - pathlib (Path)
 - typing (Dict, Iterable, List, Optional, Sequence, Tuple, Union)
 - concurrent.futures (ThreadPoolExecutor, as_completed)
 - tqdm
 - cv2
 - numpy
 - imageio.v3 (or PIL) 
 - slideio

## Example run command: 
python VSI_Tile_Cropper_SlideIO.py  --batch-dir "path/to/vsi/files" "path/to/output/directory" --tile 2400 2400 --overlap 360 360

## All Input Arguments:

    # Input / output paths
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


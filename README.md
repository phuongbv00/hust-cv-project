# HUST CV Project

## Prerequisites

- Python 3.11+
- uv: https://docs.astral.sh/uv/

## Quick start

```
uv sync
```

## Project layout

- pyproject.toml — dependencies, Python version
- p1/main.py — main entry point to process a single image
- p1/utils.py — helpers for IO and image saving
- p1/input/ — sample input images (PNG)
- p1/output/ — generated folders with intermediate results per input image
- p1/test.py — simple runner to process all PNGs in p1/input

## Run the main script

Process a single image and print the estimated grain count to stdout. Intermediate images are saved under p1/output/<
image_stem>/.

```
python -m p1.main p1/input/1_wIXlvBeAFtNVgJd49VObgQ.png
```

- Exit codes:
    - 0: success
    - 1: wrong usage (missing argument)
    - 2: runtime error (e.g., file not found)

Outputs are written to:

- p1/output/1_wIXlvBeAFtNVgJd49VObgQ/
    - 01_median.png
    - 02a_fft_magnitude.png
    - 02b_fft_mask.png
    - 02c_fft_denoised.png
    - 03_clahe.png
    - 04_blur.png
    - 05_thresh.png
    - 06_opened.png
    - 07_closed.png
    - 08_sure_bg.png
    - 09_dist.png
    - 10_sure_fg.png
    - 11_unknown.png
    - 12_markers.png
    - 13_watershed_boundaries.png

The final printed number is the component count after watershed, with a small area filter.

## Run the batch test

There is a simple test runner that processes all PNG files in p1/input.

```
python -m p1.test
```

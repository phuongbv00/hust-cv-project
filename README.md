# HUST CV Project

## Prerequisites

- Python 3.11+
- uv: https://docs.astral.sh/uv/

## Quick start

```sh
uv sync
```

## Project 1: Rice grains counting

### Prepare the dataset

```sh
curl -L -o ./p1/data.zip\
  https://www.kaggle.com/api/v1/datasets/download/phuongbv00/rice-grains-counting-samples
```

```sh
unzip ./p1/data.zip -d ./p1/
```

### Run the program

```sh
uv run python -m p1.main <image_path>
```

### Run the sample test

```sh
uv run python -m p1.test
```

## Project 2: Object detection using local features

### Prepare the dataset

```sh
curl -L -o ./p2/data.zip\
  https://www.kaggle.com/api/v1/datasets/download/phuongbv00/flickr-logos-27-augmented-for-mini-detection
```

```sh
unzip ./p2/data.zip -d ./p2/
```

#### Generate augmented images (optional)

```sh
uv run python -m p2.augment
`````

### Run the program

```sh
uv run python -m p2.main <template_image> <scene_image1> [scene_image2 ...]
```

### Run the sample test

```sh
uv run python -m p2.test
```

### Run the evaluation

```sh
uv run python -m p2.evaluate
```
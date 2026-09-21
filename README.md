# MNIST Digit Classifier

A small PyTorch program that trains a convolutional neural network (CNN) on the MNIST
handwritten-digit dataset and then uses it to count how many times each digit appears
in a folder of images.

## What it does

1. **Trains a CNN** on MNIST if no trained model exists (`digit_classifier.pth`).
   The dataset is downloaded automatically to `./data` on the first run.
   The model architecture lives in `CNN.py` (3 convolutional blocks followed by a
   fully-connected classifier).
2. **Classifies every image** in the `digits/` folder (batch of 28x28 JPGs) using the
   trained model.
3. **Counts digit occurrences** and writes the result to `output.txt` as a 10-element
   array `[d0, d1, ..., d9]`, where each element is the number of images classified as
   that digit, e.g. `[1220, 1403, 950, 1296, 1204, 1160, 1171, 695, 1266, 1635]`.

## Project structure

- `main.py` — entry point: training, inference, and counting logic
- `CNN.py` — the CNN model definition
- `digits/` — input images to classify
- `data/` — MNIST dataset (downloaded automatically)
- `digit_classifier.pth` — saved trained model
- `output.txt` — counting result
- `requirements.txt` — Python dependencies

## How to launch

### Create a virtual environment and required install packages

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

### Start the program

```bash
python main.py
```

## Notes

- A GPU is used automatically when available (CUDA), otherwise the CPU is used.
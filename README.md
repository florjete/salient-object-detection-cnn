# Salient Object Detection CNN

From-scratch PyTorch salient object detection project on MSRA10K. The official demo version is an improved encoder-decoder CNN with BatchNorm and skip connections. It does not use pretrained models or `segmentation_models_pytorch`.

## Demo Results

These are presented as an improved from-scratch baseline, not as state-of-the-art SOD results.

The final demo uses threshold `0.40` because it had the best validation F1 during threshold search.

| Version | IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.4262 | 0.6043 | 0.6356 | 0.6176 |
| Improved | 0.6470 | 0.7582 | 0.8267 | 0.7895 |

Improved test loss: `0.4917`.

## Project Structure

```text
salient-object-detection-cnn/
├── notebooks/
│   └── demo_notebook.ipynb
├── src/
│   ├── organize_dataset.py
│   ├── data_loader.py
│   ├── sod_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── outputs/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

This project uses the MSRA10K Salient Object Detection dataset.

Dataset source: [Kaggle - Saliency Object Segmentation MSRA10K](https://www.kaggle.com/datasets/evvalaycan/saliency-object-segmentation-msra10k)

The dataset is not included in this repository because it is too large for normal GitHub source control. Download it from Kaggle and store it locally or in Google Drive.

Expected structure:

```text
sod_data/
└── MSRA10K/
    ├── images/
    └── masks/
```

Where:

- `images/` contains RGB input images (`.jpg`)
- `masks/` contains corresponding saliency masks (`.png`)

Matching files should share the same base filename:

```text
101.jpg <-> 101.png
```

Google Colab users can store the dataset in Google Drive at:

```text
/content/drive/MyDrive/sod_data/MSRA10K/
```

Project Drive folder:

[sod_data Google Drive folder](https://drive.google.com/drive/folders/1f-fPQoQMt7RQ5i0HX3st4oR8P5otLv08?usp=drive_link)

Google Drive project layout:

```text
MyDrive/
└── sod_data/
    ├── MSRA10K/
    │   ├── images/
    │   └── masks/
    ├── checkpoints/
    │   └── best_model.pth
    └── outputs/
```

## Model

The official model is `ImprovedSODNet` in `src/sod_model.py`.

- Input: RGB image resized to `128x128`
- Preprocessing: image tensor values in `0..1`
- Mask preprocessing: nearest-neighbor resize, binary `0/1`
- Encoder: convolution, BatchNorm, ReLU, MaxPool
- Decoder: ConvTranspose upsampling, skip connections, convolution, BatchNorm, ReLU
- Output: one-channel sigmoid saliency mask

The skip connections are the main architectural improvement over the original baseline because they preserve spatial detail for mask boundaries.

## Loss And Metrics

Training loss:

```text
BCE + 0.5 * (1 - IoU)
```

Metrics:

- IoU
- Precision
- Recall
- F1

Evaluation supports validation threshold search. The official demo threshold is:

```text
0.40
```

## Colab Setup

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/salient-object-detection-cnn.git
cd salient-object-detection-cnn
pip install -r requirements.txt
```

## Official Demo

Open:

```text
notebooks/demo_notebook.ipynb
```

The demo notebook loads:

```text
/content/drive/MyDrive/sod_data/checkpoints/best_model.pth
```

and shows:

- input image
- ground truth mask
- predicted mask
- overlay
- inference time per image

## Evaluate Existing Checkpoint

Use the fixed official threshold:

```bash
python src/evaluate.py test \
  --data-root "/content/drive/MyDrive/sod_data/MSRA10K" \
  --checkpoint "/content/drive/MyDrive/sod_data/checkpoints/best_model.pth" \
  --threshold 0.40 \
  --visualize
```

Or rerun validation threshold search before testing:

```bash
python src/evaluate.py test \
  --data-root "/content/drive/MyDrive/sod_data/MSRA10K" \
  --checkpoint "/content/drive/MyDrive/sod_data/checkpoints/best_model.pth" \
  --search-threshold \
  --visualize
```

## Single-Image Inference

```bash
python src/evaluate.py infer \
  --image "/content/drive/MyDrive/sod_data/MSRA10K/images/101.jpg" \
  --checkpoint "/content/drive/MyDrive/sod_data/checkpoints/best_model.pth" \
  --output "/content/drive/MyDrive/sod_data/outputs/101_overlay.png" \
  --threshold 0.40
```

## Train From Scratch

This will create or overwrite the checkpoint names you pass, so use separate names if you want to preserve the official demo checkpoint.

```bash
python src/train.py \
  --data-root "/content/drive/MyDrive/sod_data/MSRA10K" \
  --checkpoint-dir "/content/drive/MyDrive/sod_data/checkpoints" \
  --output-dir "/content/drive/MyDrive/sod_data/outputs" \
  --epochs 25 \
  --batch-size 16
```

Training includes validation-loss early stopping with default patience `5`. To resume from the last saved checkpoint:

```bash
python src/train.py \
  --data-root "/content/drive/MyDrive/sod_data/MSRA10K" \
  --checkpoint-dir "/content/drive/MyDrive/sod_data/checkpoints" \
  --output-dir "/content/drive/MyDrive/sod_data/outputs" \
  --epochs 25 \
  --resume
```

## Notes

- The trained checkpoint is intentionally ignored by Git.
- If your submission requires the trained model file in GitHub, upload it as a GitHub Release asset or provide the Google Drive checkpoint link in the report.
- Dataset files are intentionally ignored by Git.
- Keep large artifacts in Google Drive.
- Use `num_workers=0` in Colab notebooks for stable DataLoader behavior.

# Azure Computer Vision – Intel Scene Classification Benchmark

This project evaluates **Azure AI Vision (Image Analysis 4.0)** on the
[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
dataset (~14k natural scene images: buildings, forest, glacier, mountain, sea, street).

## Project Goals

- Use a **real public dataset** (Intel scenes) to benchmark Azure's pre-trained vision model.
- Run **batch image analysis at scale** (14k images) using Azure Vision.
- Compute simple but meaningful metrics:
  - How often do Azure tags semantically match the ground-truth scene label?
- Build a **resume-friendly, resume-capable pipeline** that avoids double-charging for the same image.

## Tech Stack

- Python, `azure-ai-vision-imageanalysis`, `pandas`, `tqdm`
- Azure AI Vision (Image Analysis 4.0, tags + captions)
- Dataset: Intel Scene Classification (Kaggle)

## Pipeline

1. **Download dataset** to `data/raw/intel/`:
   - Place `seg_train` under `data/raw/intel/seg_train`.
2. **Configure Azure Vision**:
   - Create Azure AI Vision resource in `East US`.
   - Put `VISION_ENDPOINT` and `VISION_KEY` in `.env`.
3. **Run batch analysis**:
   - First sample run (optional):

     ```bash
     python -m src.batch_analyze   # with max_images=1000 in __main__
     ```

   - Full run (14k images, resume-safe):

     ```bash
     python -m src.batch_analyze   # with max_images=None in __main__
     ```

4. **Evaluate results**:

   ```bash
   python -m src.evaluate

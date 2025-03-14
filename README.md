# Training Image-to-Image Model: Pix2PixHD

## Project Overview

1- Creating a synthetic paired dataset of bearded and clean-shaven images
2- Training an *image*-to-image (img2img) generative model.

---

## Setting Up the Environment

Ensure you have installed before proceeding.

```bash
conda create -n img2img python=3.10
conda activate img2img
pip install -r requirements.txt
```

---

## Generating Paired Data

Run the paired data generation script as a module:

```bash
python -m utils.generate_paired_data
```

### Configuration

Modify parameters and save paths in `config.yaml`:

```yaml
paired_data:
  base_image_dir: "dataset-test/test/base_image"
  beared_dir: "dataset-test/test/beared"
  mask_dir: "dataset-test/test/mask"
  predictor_path: "shape_predictor_68_face_landmarks.dat"
  positive_prompt: "A high-resolution portrait of a clean-shaven man, looking directly at the camera, photographed in natural light."
  negative_prompt: "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w, worst quality, low quality, off-center, blurred eyes"
  model_id: "SG161222/RealVisXL_V5.0"
  num_images: 2
  width: 512
  height: 512

inpainting:
  prompt_beard: "full, thick, and natural beard that shows individual hair strands"
  num_inference_steps: 50
  guidance_scale: 10
  strength: 0.99

dataset:
  base_dir: "dataset/train/base_image"
  beard_dir: "dataset/train/beared"
  mean: [0.55, 0.48, 0.42]
  std: [0.22, 0.21, 0.21]
```

---

## Training the Pix2PixHD Model

Run the training module:

```bash
python -m training.training_pix2pixhd
```

### Training Configuration

Modify training parameters in `config.yaml`:

```yaml
training:
  batch_size: 64
  num_epochs: 100
  use_perceptual_loss: true
  perceptual_weight: 0.1
  lr: 0.0002
  lr_step: 30
  lr_gamma: 0.5
  l1_weight: 100
  checkpoint_dir: "checkpoints/experiment_2"
  checkpoint_interval: 10
  device_ids: [0, 1]
```

## Notes

- Ensure paths in `config.yaml` are correctly set before running scripts.
- Training requires a GPU for optimal performance.
- Modify hyperparameters in `config.yaml` as needed to optimize training outcomes.


# Final project — Damaged photo restoration

Practical code: [photo_restoration.ipynb](photo_restoration.ipynb). The test metrics from that run are in [test_metrics.csv](test_metrics.csv).

This project was done with **Deniz Mostafazadeghan** and **Sajjad Taghizade**.

Old photographs in this set are damaged by noise, scratches, stains, blur, and faded color. The task is supervised: each sample is a damaged image paired with a clean image, from [joshuachin/openphoto-restore-dataset](https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset) on the Hugging Face Hub. The model maps the damaged image to the clean one.

## Data

Both images in a pair are resized to 256×256 and scaled from `[0, 255]` to `[0, 1]`, so they stay aligned. The dataset’s train/test split is kept. Ten percent of the training split is held out for validation: **4,050 / 450 / 500**. The loader uses batch size 16.

## Model

The network is a U-Net, chosen over a plain autoencoder because the skip connections carry edges and texture around the bottleneck, which a restoration output needs if it is to stay aligned with the input. It is fully convolutional and trained with a pixel loss, so it does not need an adversarial loop or a sampled latent code. The training set is a few thousand pairs, which is enough for this size of U-Net.

| Block | Detail |
| --- | --- |
| Encoder | 4 stages, each two 3×3 convolutions with batch normalization and ReLU, then 2×2 max-pooling. Channels 64 → 128 → 256 → 512 |
| Bottleneck | Double convolution at 1024 channels |
| Decoder | 4 transposed-convolution stages. Each one concatenates the matching encoder map, then a double convolution |
| Head | 1×1 convolution to 3 channels, then a sigmoid, so the output sits in `[0, 1]` with the target |

The parameter count in the notebook is **31,043,651**.

## Training

The loss is L1. It is less prone to the blur that mean-squared error produces on this kind of target. The optimizer is Adam at learning rate 10⁻⁴, for 30 epochs, on a Colab GPU. Training and validation loss both fall and stay close. A single validation spike near epoch 18 is one noisy batch, not a change in the fit. The checkpoint with the best validation loss is the one evaluated below.

## Test set

|  | MSE | MAE | PSNR (dB) | SSIM |
| --- | --- | --- | --- | --- |
| U-Net | 0.0080 | 0.0579 | 22.99 | 0.863 |
| Required | — | — | ≥ 20 | ≥ 0.65 |

Both required floors are cleared.

The pixel loss is what the model is good at. Speckle, scratches, and stains are reduced, and fine structure that is faint in the damaged input (fur, scales, foliage, building edges) comes back. Color is the weak part. Outputs stay closer to the muted damaged input than to the saturated ground truth, including on the underwater and sunset examples. L1 on an under-determined color mapping rewards an average color when many colorings would fit the same damaged input. Structure is less ambiguous, which is why SSIM, which weights structure more than exact color, clears its threshold by more than the color match would suggest.

The same limit is why a single L1 term cannot represent several plausible restorations of one input. The run is scored only with full-reference metrics, at a fixed 256×256. A perceptual or adversarial term next to L1, augmentation, a learning-rate schedule past epoch 30, and a check at higher resolution are the changes the report points to. The loss was still decreasing slowly at epoch 30.

# Homework 3 — A cascade detector and a colorization network

Two notebooks, both using weights that were already trained.

| Notebook | Task |
| --- | --- |
| [cat_detection.ipynb](cat_detection.ipynb) | Frontal cat faces with a Haar cascade |
| [colorization.ipynb](colorization.ipynb) | Grayscale to color with a convolutional autoencoder in Lab |

## Cat faces

The detector is OpenCV’s `haarcascade_frontalcatface_extended.xml`, called as `CascadeClassifier.detectMultiScale` with `scaleFactor=1.1`, `minNeighbors=5`, and `minSize=(50, 50)`. Detection runs on a blurred grayscale image. A stronger blur, `(9, 9)`, collapses nested boxes that come from detections at several scales, so one cat keeps one box. The drawn label is `Cat`.

The cascade only reads brightness. Feeding it the color frame or the grayscale frame produces the same boxes, because both paths are converted to grayscale and blurred before detection. A 90° rotation breaks it: the cascade was trained on upright frontal faces, and after the rotation the detections drop, shift, or disappear. Covering that case needs several rotated copies of the image, or a detector that was trained with rotation.

## Colorization

The network is `colorize_autoencoder_first1000.keras`. The input is the L channel of Lab at 256×256, with values in `[0, 100]`. The output is the a and b channels, passed through `tanh`, so they lie in `[-1, 1]` and are scaled by 110 before `lab2rgb`. The encoder uses max-pooling and the decoder uses upsampling.

Two ways of building L are compared on four images.

1. Repeat the grayscale channel three times, convert with `rgb2lab`, and keep L.
2. Let `cv2.imread(..., IMREAD_COLOR)` expand the gray image to three channels, then take L the same way.

They match in practice, because both build an RGB image whose three channels are equal. Small differences come from channel order or from `uint8` versus a NumPy stack. The rest of the tests use the first method.

A 45° rotation, followed by colorization and a −45° rotation back, leaves flat regions (sky, walls) close to the unrotated result and shifts color along edges and fine texture (leaves, buildings). Empty corners and diagonal edges from the rotation are the cause. The network was not trained on rotated 256×256 inputs.

Gaussian noise with standard deviation 0.1, added to L before the resize and the clip to `[0, 100]`, hits steep gradients and fine texture first (tree edges, building lines, the sky boundary). Smooth regions stay more stable, because a and b are predicted from L and the noise is sitting on L.

Scaling L by 0.7 darkens the image while a and b stay close to the normal run: the network reads color from the structure of L, not from the brightness level alone. Scaling by 1.3 and clipping at 100 washes out the bright areas, because the highlight detail is gone, and the new colors do not simply follow the brightness change.

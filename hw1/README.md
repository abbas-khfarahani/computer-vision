# Homework 1 — Image processing

Practical code: [image_processing.ipynb](image_processing.ipynb). It converts one photograph through RGB, grayscale, binary, and HSV, measures contrast, and applies histogram equalization and stretching. The second part adds salt-and-pepper and Gaussian noise and compares a hand-written convolution with OpenCV. The third part does the same denoising on a short video.

## Color and contrast

`Pic.jpg` loads as an 800×633 RGB image, `uint8`, three channels.

| Version | What it stores |
| --- | --- |
| RGB | Red, green, and blue at each pixel |
| Grayscale | One brightness channel |
| Binary | Two levels, here 0 and 255 |
| HSV | Hue, saturation, and value |

Contrast is the standard deviation of the grayscale intensities.

| Image | Contrast (std) |
| --- | --- |
| Original | 54.84 |
| Contrast increased | 63.65 |
| Contrast decreased | 32.91 |

Histogram equalization redistributes the intensities so the overall contrast rises. Histogram stretching maps the existing range onto the full `[0, 255]` interval.

## Filters

Salt-and-pepper noise sets random pixels to pure black or pure white. Gaussian noise shifts brightness by a normal draw.

A kernel is a small matrix that slides over the image. The notebook implements 2D convolution for any odd kernel size, with zero padding.

On the salt-and-pepper image, a 3×3 mean (convolutional low-pass) reduces the noise and also softens edges. A 3×3 median (non-convolutional low-pass) removes the isolated impulses and keeps the edges better, so the median is the better filter for this noise. Two high-pass kernels are applied to the clean image: one emphasizes edges, and the other also sharpens the whole frame.

The same filters through OpenCV (`blur`, `medianBlur`, `filter2D`, `GaussianBlur`) match the custom outputs. The custom mean is close to `blur`. Median remains the best salt-and-pepper cleaner. The custom high-pass results behave like `filter2D`.

## Video

`Original_Vid.mp4` has 309 frames at 29.97 fps, resolution 640×480, duration 10.31 seconds. Salt-and-pepper noise is added to every frame, then a median filter is applied. The median is used for the same reason as on the still image: single-pixel impulses disappear, and edges stay sharper than they would under a mean filter.

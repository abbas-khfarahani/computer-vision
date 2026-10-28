# Homework 2 — CAPTCHA characters without a learned classifier

Practical code: [captcha_ocr.ipynb](captcha_ocr.ipynb). It builds 20 four-character images, denoises them, cuts out the glyphs, and compares each glyph with a template. The labels and the logged scores are in [segmentation_report.csv](segmentation_report.csv) and [image_predictions.csv](image_predictions.csv).

## Data

Twenty strings are drawn from lowercase letters and digits and rendered with Arial at size 40. The notebook’s run uses these labels: `xaji`, `0y6d`, `pbhs`, `ahxt`, `hv3a`, `3zmf`, `8mdd`, `4v30`, `t9nt`, `3w5u`, `zbik`, `cidk`, `wnnh`, `j7xv`, `g0fn`, `9xuy`, `41ib`, `ljh7`, `5lxo`, `6qji`.

Salt-and-pepper noise (amount 0.03) is added to each image. A 3×3 mean filter blurs the strokes. A 3×3 median removes the impulses and keeps the character boundaries, so the median is the filter kept from this section.

## Preprocessing

A second denoise uses a larger neighborhood than the 3×3 kernel above: a 5×5 blur, which averages more of the neighborhood and therefore removes more of the remaining speckle, at the cost of softer edges. Sharpening is an unsharp mask, chosen because it puts those edges back after the blur without a separate deconvolution.

Binarization then drops connected components smaller than 20 pixels. The threshold is the gray level that splits foreground from background. It is not a fixed constant. Otsu sets it from each image’s histogram. In this run it is 125 on most images, 124 on `img_12`, and 126 on `img_17`.

## Segmentation

Each binary image is split into connected components, the small ones are discarded, and the rest are sorted left to right and resized to 64×64 so they match the reference templates.

Most of the 20 images yield 4 components, which is the string length. Seven images yield 5 or 6 (`img_00`, `img_10`, `img_11`, `img_13`, `img_16`, `img_17`, `img_19`), so a stroke was split or a leftover blob was kept.

## Recognition

Matching is nearest template by cosine similarity. No classifier is trained. The reference set in the course folder was not available in the run that produced these files, so the notebook built a synthetic stand-in of 36 templates. With those templates the logged character accuracy is **0%** and the exact-match accuracy is **0%**, against a 90% target. The predicted strings in `image_predictions.csv` are short and repetitive (`1`, `g`, `f`, `4`), which is what that stand-in set returns. The same notebook notes that `l`/`1` and `o`/`0` need extra cues (edge structure, the box aspect ratio, and resizing the real templates to the glyph) once the course mapset is the reference.

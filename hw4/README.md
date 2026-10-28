# Homework 4 — Intel scene classification

Practical code: [intel_scene_classification.ipynb](intel_scene_classification.ipynb).

This homework was done with **Deniz Mostafazadeghan** and **Sajjad Taghizade**.

The dataset is Intel Image Classification: 14,034 training images and 3,000 test images, RGB, resized to 150×150. The six classes are buildings, forest, glacier, mountain, sea, and street. A 20% split of the training folders, seed 123, gives 11,228 training images and 2,806 validation images. Labels are integer class indices, not one-hot vectors. Pixels are rescaled from `[0, 255]` to `[0, 1]`. Training augmentation is a horizontal flip, a rotation of 0.1, and a zoom of 0.1. Validation and test images are only rescaled.

Images stay as 150×150×3 tensors through the convolutions. Flatten is applied after the last pooling layer so the feature volume can enter the dense layer. That dense layer needs a fixed-length vector, which is why the spatial size is fixed at the input.

## Baseline

The network is the one specified for the homework, with the augmentation and the rescaling inside the model.

| Stage | Layer |
| --- | --- |
| Input | 150×150×3 |
| Augmentation | Random flip, rotation 0.1, zoom 0.1 |
| Normalization | Rescaling by 1/255 |
| Conv blocks | Conv2D 32, 64, 128, 128, each ReLU, each followed by 2×2 max-pooling |
| Head | Flatten, Dense 128 ReLU, Dense 6 softmax |

It has 1,568,838 parameters. The loss is sparse categorical cross-entropy, which matches integer labels over six classes. The optimizer is Adam. Training is 10 epochs, batch size 32.

| Epoch | Train accuracy | Train loss | Val accuracy | Val loss |
| --- | --- | --- | --- | --- |
| 1 | 0.5612 | 1.1021 | 0.6536 | 0.9428 |
| 5 | 0.7799 | 0.6094 | 0.8008 | 0.5634 |
| 10 | 0.8308 | 0.4674 | 0.8261 | 0.4996 |

At epoch 10 the accuracy gap (train minus validation) is 0.0047 and the loss gap (validation minus train) is 0.0322. The two curves stay together, so this run is not overfitting.

On the 3,000-image test set, `evaluate` reports accuracy **0.8220** and loss **0.5474**. That is above the 80% mark set for the homework.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| buildings | 0.81 | 0.73 | 0.77 | 437 |
| forest | 0.95 | 0.96 | 0.95 | 474 |
| glacier | 0.90 | 0.66 | 0.76 | 553 |
| mountain | 0.78 | 0.78 | 0.78 | 525 |
| sea | 0.76 | 0.89 | 0.82 | 510 |
| street | 0.78 | 0.92 | 0.84 | 501 |

Forest is the easy class. Glacier has the lowest recall (0.66). The confusions concentrate on the natural scenes that share texture (glacier, mountain, sea) and on the two urban classes (buildings, street).

## A deeper variant

The second network adds batch normalization after the convolutions and the dense layer, dropout 0.4 on the dense head, a wider and deeper last conv stage (256 filters, plus an extra conv block), Adam at learning rate 10⁻⁴, and 15 epochs. It has 2,032,838 parameters.

On the same test set this run reaches accuracy **0.7373** and loss **0.8393**. That is 8.5 percentage points below the baseline. The extra capacity and the smaller step size did not improve this training run. The baseline, at 0.8220, is the model that clears the homework threshold.

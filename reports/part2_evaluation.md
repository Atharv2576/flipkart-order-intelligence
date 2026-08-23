# Part 2 -- Product image categoriser evaluation

Source data: Fashion-MNIST (Zalando Research), the pinned, keyless benchmark,
loaded via `torchvision.datasets.FashionMNIST(download=True)`.

## Splits

| split | images | note |
|---|---:|---|
| train | 54,000 | 60k train half minus the validation carve-out |
| validation | 6,000 | stratified, 600/class, carved from the train half |
| test | 10,000 | canonical test split, untouched until final evaluation |

## Preprocessing

Grayscale channel replicated to 3, resized to 224x224 (ResNet-18's native
ImageNet input size), normalized with ImageNet mean/std.

## Transfer learning

Pretrained ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`); `conv1`, `bn1`,
`layer1`-`layer4` frozen; `fc` replaced with a fresh `Linear(512, 10)` and
trained via Adam (lr=1e-3, batch size 256, 12 epochs).

## Feature caching

The frozen backbone's output for a given image never changes across epochs,
so it is run once over all 70,000 images and the 512-d vectors are cached;
the head then trains on those cached tensors instead of 12 full forward
passes through ResNet-18. Measured: feature extraction took ~231s on Apple
M3 (MPS); head training on the cached vectors then took **4.6s** for all 12
epochs.

## Feature-extraction result

Validation accuracy after feature-extraction-only training: **89.80%**
(final epoch), clearing the 80% fine-tuning trigger, so `layer4` was never
unfrozen and no second training stage ran. See
`reports/part2_training_log.json` for the full per-epoch curve.

## Final test-set result

**Test accuracy: 88.75%** on the 10,000 held-out test images (untouched
until this evaluation). Full confusion matrix at
`reports/part2_confusion_matrix.csv`, per-class precision/recall/F1 at
`reports/part2_per_class_metrics.csv`.

## Confusion patterns (read directly off the matrix)

**Shirt <-> T-shirt/top -- 220 misclassifications (138 + 82).** Both are
short-to-medium-sleeved upper-body garments photographed flat against the
same background. At 28x28 source resolution, the only real distinguishing
feature -- a button placket or collar -- occupies a handful of pixels and is
frequently smoothed away by downsampling before the image ever reaches the
224x224 input the backbone sees; upsampling later cannot recover detail that
was never captured, since it only interpolates the existing 784 pixels.

**Shirt <-> Coat -- 180 misclassifications (95 + 85).** A long-sleeved shirt
and a coat share the same basic silhouette: a rectangular torso with two
sleeves of similar length. Their real-world difference -- fabric thickness,
layering, fastening hardware -- shows up as subtle intensity gradients, not
shape changes, and Fashion-MNIST is greyscale, so the colour/texture cues
(wool weave vs. cotton) a shopper would use instantly are unavailable to the
model. It is left comparing two nearly identical binary silhouettes.

**What this means for the catalogue use case.** Both confusion pairs stay
inside the same visually coherent family (upper-body apparel) and never
cross into footwear, bags, or trousers. For "is this photo filed under
roughly the right department," that is the useful failure mode: a
mis-tagged shirt still lands in apparel, so a support agent using Part 3's
tool gets the right department even on the model's worst days.

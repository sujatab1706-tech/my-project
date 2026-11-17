"""Grad-CAM: a heatmap of WHERE in the photo the CNN looked to make its decision.

How it works (simplified):
  1. MobileNetV2's last convolution layer gives 7x7 "feature maps": each is a
     detector for some pattern, with a value for each region of the photo.
  2. We ask: if this feature map got stronger, how much would the 'tlgw'
     (or 'nwb') probability go up? That's the GRADIENT.
  3. Weight each map by its gradient, add them up, keep the positive part.
     Bright regions = "this part of the image pushed the answer towards YES".

If the model flags TLGW and the heatmap lights up on the "SALE" text, great.
If it lights up somewhere random, the model may have learned a shortcut.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

LABELS = ["tlgw", "nwb"]


def _parts(model):
    base = next(l for l in model.layers if isinstance(l, keras.Model))  # the MobileNetV2 body
    head = [l for l in model.layers if l is not base and not isinstance(l, keras.layers.InputLayer)]
    return base, head


def gradcam(model, img: Image.Image, label: str):
    """Return (probability, heatmap 224x224 with values 0-1) for one label."""
    base, head = _parts(model)
    x = np.asarray(img.convert("RGB").resize((224, 224)), dtype="float32")[None]
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    with tf.GradientTape() as tape:
        conv = base(x, training=False)       # (1, 7, 7, 1280) feature maps
        tape.watch(conv)
        out = conv
        for layer in head:                   # pooling -> dropout -> dense
            out = layer(out, training=False)
        score = out[:, LABELS.index(label)]
    grads = tape.gradient(score, conv)[0]               # (7, 7, 1280)
    weights = tf.reduce_mean(grads, axis=(0, 1))        # importance of each feature map
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.BILINEAR)) / 255
    return float(score.numpy()[0]), cam


def overlay(img: Image.Image, cam: np.ndarray, alpha=0.45):
    """Blend the heatmap (red = important) on top of the photo."""
    import matplotlib.cm as cm
    heat = np.uint8(cm.jet(cam)[..., :3] * 255)
    base = np.asarray(img.convert("RGB").resize((224, 224)))
    return Image.fromarray(np.uint8(base * (1 - alpha) + heat * alpha))

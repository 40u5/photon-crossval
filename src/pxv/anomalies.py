"""Controlled model error: insert content the face prior has never seen into otherwise well-specified scenes.

Every anomaly returns (x_anomalous, mask).  Because the clean scene is drawn from the prior itself, the anomaly is
the *only* source of prior misspecification in the paired scene -- any change in the diagnostics is caused by it.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Eye centres of FFHQ-aligned 64x64 faces (x, y), estimated from prior samples.
EYES = ((23, 30), (41, 30))
KINDS = ('eye_patch', 'lesion', 'text', 'shape')


def _canvas(draw_fn, size=64):
    img = Image.new('L', (size, size), 0)
    draw_fn(ImageDraw.Draw(img))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)


def _paint(x, mask, rgb):
    """Composite a flat colour (x-space RGB) into x with soft mask (H, W)."""
    col = torch.tensor(rgb, dtype=x.dtype, device=x.device).view(3, 1, 1)
    m = mask.to(x.device)[None]
    return x * (1 - m) + col * m


def eye_patch(x, rng):
    """An opaque patch over one eye: the prior will want to paint an eye that is not there."""
    cx, cy = EYES[rng.integers(2)]
    rx, ry = rng.integers(6, 8), rng.integers(4, 6)
    mask = _canvas(lambda d: d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255))
    tone = rng.uniform(-0.9, -0.5)  # dark cloth
    return _paint(x, mask, (tone, tone, tone + 0.05)), mask


def lesion(x, rng):
    """A small dark or bright spot on the cheek or forehead: the prior will tend to erase it."""
    cx, cy = (int(rng.integers(18, 46)), int(rng.choice([rng.integers(14, 20), rng.integers(38, 46)])))
    r = int(rng.integers(2, 4))
    mask = _canvas(lambda d: d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255))
    rgb = (-0.8, -0.85, -0.85) if rng.random() < 0.5 else (0.9, 0.2, 0.9)  # dark mole / magenta mark
    return _paint(x, mask, rgb), mask


def text(x, rng):
    """A short printed word on the forehead or cheek (a sticker / tattoo)."""
    word = str(rng.choice(['SPAD', 'UofT', 'HI', '404', 'OK']))
    font = ImageFont.load_default()
    y0 = int(rng.choice([12, 44]))
    mask = _canvas(lambda d: d.text((32 - 3 * len(word), y0), word, fill=255, font=font))
    rgb = (0.95, 0.95, 0.2) if rng.random() < 0.5 else (-0.95, -0.95, -0.95)
    return _paint(x, mask, rgb), mask


def shape(x, rng):
    """A thin geometric mark (cross or ring)."""
    cx, cy = int(rng.integers(20, 44)), int(rng.integers(16, 48))
    r = int(rng.integers(4, 7))
    if rng.random() < 0.5:
        mask = _canvas(lambda d: (d.line([cx - r, cy, cx + r, cy], fill=255, width=2),
                                  d.line([cx, cy - r, cx, cy + r], fill=255, width=2)))
    else:
        mask = _canvas(lambda d: d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=2))
    rgb = (0.1, 0.9, 0.1) if rng.random() < 0.5 else (-0.9, -0.9, 0.6)
    return _paint(x, mask, rgb), mask


def _region_mask(x0, y0, w, h, size=64):
    m = torch.zeros(size, size)
    m[y0:y0 + h, x0:x0 + w] = 1
    return m


def checker(x, rng, bin=4, amp=0.45):
    """Null-space anomaly for a `bin`-binned SPAD: a 1-pixel checkerboard with *exactly zero* change of every
    SPAD-pixel mean.  Obvious to the eye, invisible to the sensor -- no held-out photon can ever contradict it."""
    x0, y0 = bin * int(rng.integers(3, 9)), bin * int(rng.choice([3, 10]))   # on the forehead or a cheek
    mask = _region_mask(x0, y0, 4 * bin, 3 * bin)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    pattern = (amp * (1 - 2 * ((xx + yy) % 2)) * mask).to(x.device)
    xa = x + pattern
    for _ in range(3):  # keep every bin x bin block mean unchanged despite clipping to [-1, 1]
        xa = xa.clamp(-1, 1)
        delta = torch.nn.functional.avg_pool2d((x - xa)[None], bin)[0]
        xa = xa + torch.nn.functional.interpolate(delta[None], scale_factor=bin)[0] * mask.to(x.device)
    return xa, mask


def isolum(x, rng, w=(0.2126, 0.7152, 0.0722), amp=0.6):
    """Null-space anomaly for a monochrome SPAD: a colour patch with unchanged luminance."""
    x0, y0 = int(rng.integers(16, 36)), int(rng.choice([12, 40]))
    mask = _region_mask(x0, y0, 12, 10).to(x.device)
    chroma = torch.tensor((1.0, -w[0] / w[1], 0.0) if rng.random() < 0.5 else (0.0, -w[2] / w[1], 1.0),
                          device=x.device).view(3, 1, 1)
    wv = torch.tensor(w, device=x.device).view(3, 1, 1)
    xa = x + amp * chroma * mask
    for _ in range(3):  # restore the luminance after clipping by a grey offset
        xa = xa.clamp(-1, 1)
        xa = xa + ((x - xa) * wv).sum(0, keepdim=True) * mask
    return xa, mask.cpu()


def insert(x, kind, rng):
    fns = dict(eye_patch=eye_patch, lesion=lesion, text=text, shape=shape, checker=checker, isolum=isolum)
    return fns[kind](x, rng)

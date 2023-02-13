import numpy as np
from PIL import Image


def show_hdr_fromarray(img, tone_mapping='clip'):
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

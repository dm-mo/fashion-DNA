import base64
from functools import cache

import PIL
import torchvision

__all__ = ['tensor_to_image', 'tensor_to_url', 'pil_to_tensor', 'square_padding']

tensor_to_image = torchvision.transforms.ToPILImage()
pil_to_tensor = torchvision.transforms.PILToTensor()


@cache
def tensor_to_url(tensor, size=128):
    return fr"data:image/png;base64,{base64.b64encode(PIL.ImageOps.contain(tensor_to_image(tensor), (size, size))._repr_png_()).decode('ascii')}"


def square_padding(img):
    h, w = img.shape[1:]
    if h != w:
        new_w = max(h, w)
        pad_h, rem_h = divmod(new_w - h, 2)
        pad_w, rem_w = divmod(new_w - w, 2)
        padding = [pad_w, pad_h, pad_w + rem_w, pad_h + rem_h]
        return torchvision.transforms.functional.pad(img, padding, padding_mode='edge')
    return img

from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw

from .rmqrcode import rMQR


class QRImage:
    def __init__(self, qr: rMQR, module_size: int = 10):
        self._module_size = module_size
        qr_list = qr.to_list()
        self._img = Image.new("RGB", (len(qr_list[0]) * module_size, len(qr_list) * module_size), (255, 255, 255))
        self._make_image(qr_list)

    def show(self):
        self._img.show()
        pass

    def get_ndarray(self):
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is not installed")

        return np.array(self._img)

    def save(self, name: Union[str, bytes, Path]):
        self._img.save(name)

    def _make_image(self, qr_list: list[list[int]]):
        draw = ImageDraw.Draw(self._img)
        for y in range(len(qr_list)):
            for x in range(len(qr_list[0])):
                r, g, b = (0, 0, 0) if qr_list[y][x] else (255, 255, 255)
                draw.rectangle(
                    xy=(
                        x * self._module_size,
                        y * self._module_size,
                        (x + 1) * self._module_size,
                        (y + 1) * self._module_size,
                    ),
                    fill=(r, g, b),
                )

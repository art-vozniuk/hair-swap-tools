from PIL import Image
from typing import Tuple

import hair_transfer.infer_full as hair_transfer_api

def hair_transfer(source_image: Image.Image, reference_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    return hair_transfer_api.hair_transfer(source_image, reference_image)
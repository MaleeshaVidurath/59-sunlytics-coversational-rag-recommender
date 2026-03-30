# 1. Import the global data loader
from shared.data_loader import data_loader

# 2. Call get_image() with the article ID string
image_path = data_loader.get_image('0108775044')

print(f"The image is saved locally at: {image_path}")

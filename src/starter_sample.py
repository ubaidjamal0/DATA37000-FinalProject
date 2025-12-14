#%%
# %pip install openimages
# 
# %%
# Download Open Images subset
import os
from dataset_utils import download_openimages_subset, preview_imagefolder

CLASSES = ["Cat", "Dog", "Horse", "Bird", "Sheep"]
IMAGES_PER_CLASS = 300
MAX_SAMPLES = 15000

output_dir = download_openimages_subset(
    classes=CLASSES,
    max_samples=MAX_SAMPLES,
    export_dir=f"..{os.sep}data{os.sep}bigdata{os.sep}open_images_5animals_300each",
    split="train",
    dataset_name="five_animals_300each"
)

preview_imagefolder(output_dir)
print("Download Done. ImageFolder at:", output_dir)


#%%



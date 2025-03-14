import os
from PIL import Image
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(self, base_dir: str, beard_dir: str, transform=None):
        self.transform = transform
        
        self.base_files = {}
        for f in os.listdir(base_dir):
            if f.startswith("base_image_") and f.endswith('.png'):
                idx = f.replace("base_image_", "").replace(".png", "")
                self.base_files[idx] = os.path.join(base_dir, f)
                
        self.beard_files = {}
        for f in os.listdir(beard_dir):
            if f.startswith("inpainted_beard_") and f.endswith('.png'):
                idx = f.replace("inpainted_beard_", "").replace(".png", "")
                self.beard_files[idx] = os.path.join(beard_dir, f)
                
        self.common_indices = sorted(
            list(set(self.base_files.keys()).intersection(set(self.beard_files.keys()))),
            key=lambda x: int(x)
        )
        if len(self.common_indices) == 0:
            raise ValueError("No paired images found. Check naming conventions or file paths.")
        
    def __len__(self):
        return len(self.common_indices)
    
    def __getitem__(self, idx):
        index = self.common_indices[idx]
        base_path = self.base_files[index]
        beard_path = self.beard_files[index]
        
        base_img = Image.open(base_path).convert("RGB")
        beard_img = Image.open(beard_path).convert("RGB")
        
        if self.transform:
            base_img = self.transform(base_img)
            beard_img = self.transform(beard_img)
            
        return beard_img, base_img

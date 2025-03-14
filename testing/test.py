import torch
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pix2pixhd import GlobalGenerator
import matplotlib.pyplot as plt

def load_state_dict_no_module(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    return model

def test_model(input_image_path, checkpoint_path, output_image_path=None, device="cuda", visualize=False):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.55, 0.48, 0.42], std=[0.22, 0.21, 0.21])
    ])
    
    image = Image.open(input_image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    netG = GlobalGenerator(input_nc=3, output_nc=3)
    netG = load_state_dict_no_module(checkpoint_path, netG, device)
    netG.to(device)
    netG.eval()
    
    with torch.no_grad():
        output_tensor = netG(input_tensor)
    
    output_tensor = (output_tensor + 1) / 2
    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())
    
    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title("Input Bearded Image")
        axs[0].axis("off")
        axs[1].imshow(output_image)
        axs[1].set_title("Predicted Clean-Shaved Image")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        if output_image_path is not None:
            output_image.save(output_image_path)
            print(f"Inference complete. Output saved to {output_image_path}.")
        else:
            print("Output image path not provided. Skipping saving.")


if __name__ == "__main__":
    input_image = "dataset/test/beared/inpainted_beard_5.png"
    checkpoint = "checkpoints/experiment_1/netG_epoch_100.pth"
    output_image = "outputs/test_output.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model(input_image, checkpoint, output_image, device)

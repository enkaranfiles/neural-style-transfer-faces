import os
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pix2pixhd import GlobalGenerator, NLayerDiscriminator
from util.dataset import PairedDataset
from util.config import load_config  

from .perceptual_loss import VGGPerceptualLoss  


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

config_path = "config.yaml"
project_root = os.path.dirname(os.path.abspath(config_path))

def train_model(config):
    beard_dir = os.path.join(project_root, config["dataset"]["beard_dir"])
    base_dir  = os.path.join(project_root, config["dataset"]["base_dir"])
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    n_blocks = config["training"].get("n_blocks", 9)
    checkpoint_dir = os.path.join(project_root, config["training"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["dataset"]["mean"], std=config["dataset"]["std"])
    ])

    dataset = PairedDataset(beard_dir=beard_dir, base_dir=base_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = GlobalGenerator(input_nc=3, output_nc=3, n_blocks=n_blocks).to(device)
    netD = NLayerDiscriminator(input_nc=3).to(device)
    
    if torch.cuda.device_count() > 1:
        device_ids = config["training"].get("device_ids", [0, 1])
        netG = nn.DataParallel(netG, device_ids=device_ids)
        netD = nn.DataParallel(netD, device_ids=device_ids)
    
    criterionGAN = nn.MSELoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    
    
    use_perceptual_loss = config["training"].get("use_perceptual_loss", False)
    if use_perceptual_loss:
        percep_criterion = VGGPerceptualLoss().to(device)
        percep_weight = config["training"].get("perceptual_weight", 0.1)
    else:
        percep_criterion = None
        percep_weight = 0.0

    optimizerG = optim.Adam(netG.parameters(), lr=config["training"]["lr"], betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=config["training"]["lr"], betas=(0.5, 0.999))
    
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=config["training"]["lr_step"], gamma=config["training"]["lr_gamma"])
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=config["training"]["lr_step"], gamma=config["training"]["lr_gamma"])
    
    loss_D_history = []
    loss_G_history = []
    
    for epoch in range(num_epochs):
        epoch_loss_D = []
        epoch_loss_G = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for beard, base in pbar:
            beard = beard.to(device)
            base = base.to(device)
            
            optimizerD.zero_grad()
            pred_real = netD(base)
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real, device=device))
            
            fake = netG(beard)
            pred_fake = netD(fake.detach())
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake, device=device))
            
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizerD.step()
            

            optimizerG.zero_grad()
            
            pred_fake = netD(fake)
            loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake, device=device))
            
            
            loss_G_L1 = criterionL1(fake, base)
            
            
            if use_perceptual_loss and percep_criterion is not None:
                loss_G_perc = percep_criterion(fake, base)
            else:
                loss_G_perc = torch.zeros(1, device=device)
            
            
            loss_G = loss_G_GAN \
                     + config["training"]["l1_weight"] * loss_G_L1 \
                     + percep_weight * loss_G_perc
            
            loss_G.backward()
            optimizerG.step()
            
            epoch_loss_D.append(loss_D.item())
            epoch_loss_G.append(loss_G.item())
            pbar.set_postfix({
                "Loss_D": f"{loss_D.item():.4f}",
                "Loss_G": f"{loss_G.item():.4f}",
                "PercLoss": f"{loss_G_perc.item():.4f}" if use_perceptual_loss else 0.0
            })
        
        loss_D_history.append(epoch_loss_D)
        loss_G_history.append(epoch_loss_G)
        
        schedulerG.step()
        schedulerD.step()
        logging.info(f"Epoch {epoch+1} complete. LR_G: {schedulerG.get_last_lr()[0]:.6f}, LR_D: {schedulerD.get_last_lr()[0]:.6f}")
        
        if (epoch + 1) % config["training"]["checkpoint_interval"] == 0:
            torch.save(netG.state_dict(), os.path.join(checkpoint_dir, f"netG_epoch_{epoch+1}.pth"))
            torch.save(netD.state_dict(), os.path.join(checkpoint_dir, f"netD_epoch_{epoch+1}.pth"))
            logging.info(f"Saved checkpoints at epoch {epoch+1}.")

    loss_history = {"loss_D": loss_D_history, "loss_G": loss_G_history}
    torch.save(loss_history, os.path.join(checkpoint_dir, "loss_history.pth"))
    logging.info("Training complete! Loss history saved.")


if __name__ == "__main__":
    config = load_config(config_path)
    train_model(config)

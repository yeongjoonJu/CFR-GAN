import random
import torch
from torch.utils.tensorboard import SummaryWriter

class CFRLogger(SummaryWriter):
    def __init__(self, logdir):
        super(CFRLogger, self).__init__(logdir)
    
    def log_training(self, loss_D, loss_GAN, loss_mask, loss_per, loss_rec, loss_id, iteration):
        self.add_scalars('GAN_loss', {"D": loss_D, "G": loss_GAN}, global_step=iteration)
        self.add_scalar('Occ mask', loss_mask, iteration)
        self.add_scalar("Perceptual loss", loss_per, iteration)
        self.add_scalar("Identity loss", loss_id, iteration)
        self.add_scalar("Occlusion-aware Rec loss", loss_rec, iteration)

    def log_training_abl(self, loss_D, loss_GAN, loss_per, loss_rec, iteration):
        self.add_scalars('GAN_loss', {"D": loss_D, "G": loss_GAN}, global_step=iteration)
        self.add_scalar("Perceptual loss", loss_per, iteration)
        self.add_scalar("Occlusion-aware Rec loss", loss_rec, iteration)
    
    def log_train_image(self, rotated_grid, guidance_grid, out_grid, occ_grid, target_grid, iteration):
        self.add_image("input", rotated_grid, iteration)
        self.add_image("guidance", guidance_grid, iteration)
        self.add_image("output", out_grid, iteration)
        self.add_image("occlusion", occ_grid, iteration)
        self.add_image("target", target_grid, iteration)

    def log_train_image_abl(self, rotated_grid, guidance_grid, out_grid, target_grid, iteration):
        self.add_image("input", rotated_grid, iteration)
        self.add_image("guidance", guidance_grid, iteration)
        self.add_image("target", target_grid, iteration)
        self.add_image("output", out_grid, iteration)
    
    def log_test_image(self, img_grid, out_grid, rot_grid, gui_grid, mask_grid, iteration):
        self.add_image("test-input", img_grid, iteration)
        self.add_image("test-output", out_grid, iteration)
        self.add_image("test-rotated", rot_grid, iteration)
        self.add_image("test-guidance", gui_grid, iteration)
        self.add_image("test-occlusion", mask_grid, iteration)

    def log_test_image_abl(self, img_grid, out_grid, rot_grid, gui_grid, iteration):
        self.add_image("test-input", img_grid, iteration)
        self.add_image("test-output", out_grid, iteration)
        self.add_image("test-rotated", rot_grid, iteration)
        self.add_image("test-guidance", gui_grid, iteration)
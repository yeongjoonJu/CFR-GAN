import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import random
import torch.nn.functional as F

class CFRDataset(Dataset):
    def __init__(self, img_path, cfr_path, filelist, test=False, img_size=224):
        self.img_path = img_path
        self.cfr_path = cfr_path
        self.test = test
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
        ])

        self.img_list = []
        with open(filelist, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            filename = line.strip()
            self.img_list.append(filename)
            # if not self.test:
            #     self.img_list.append(filename[:-4]+'.1.jpg')
        
        print('The number of data: {}'.format(len(self.img_list)))
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        splited = self.img_list[index].split('.')
        addition=""
        # if splited[1]!='jpg':
        #     addition=".1"
            
        guidance = Image.open(self.cfr_path+'/'+splited[0]+'_gui'+addition+'.jpg')
        rotated = Image.open(self.cfr_path+'/'+splited[0]+'_rot'+addition+'.jpg')                
        img = Image.open(os.path.join(self.img_path, splited[0] + '.jpg')).convert('RGB')
        
        img = self.transform(img)
        rotated = self.transform(rotated)
        guidance = self.transform(guidance)

        if self.test:
            return img, rotated, guidance

        occ_mask = Image.open(self.cfr_path+'/'+splited[0]+'_occ'+addition+'.jpg')
        occ_mask = self.transform(occ_mask)
        # back_mask = self.transform(back_mask)
        # back_mask = blur(1.-back_mask, cuda=False).squeeze(0)
        # img[back_mask.repeat(3,1,1)>0.9] *= 0.5
        # img = img * back_mask

        if torch.rand(1) < 0.25:
            rotated = F.interpolate(rotated.unsqueeze(0), (self.img_size//2, self.img_size//2), mode='bilinear', align_corners=True)
            rotated = F.interpolate(rotated, (self.img_size,self.img_size), mode='bilinear', align_corners=True).squeeze(0)
        
        mask = torch.mean(guidance, dim=0, keepdim=True)
        mask[mask > 0.03] = 1.0
        mask[mask < 1.0] = 0.0

        # Erasing
        if torch.rand(1) < 0.4:
            # Box
            if torch.rand(1) < 0.5:
                # x y w h
                x = random.randint(int(self.img_size*0.3),int(self.img_size*0.75))
                y = random.randint(int(self.img_size*0.3),int(self.img_size*0.7))
                w = random.randint(5, int(self.img_size*0.33))
                h = random.randint(5, int(self.img_size*0.33))

                # Choose types of box
                if torch.rand(1) < 0.5:
                    random_box = -1 * torch.rand(3,1,1) + 1
                else:
                    random_box = torch.mean(rotated[:,y:y+h,x:x+w], dim=[1,2], keepdim=True)
                    
                # Give random noise
                if torch.rand(1) < 0.5:
                    noise = -0.5*torch.rand(3,self.img_size,self.img_size) + 0.25
                    random_box = (random_box.expand_as(rotated) + noise) * mask
                else:
                    random_box = random_box.expand_as(rotated) * mask

                rotated[:,y:y+h,x:x+w] = random_box[:,y:y+h,x:x+w]
                occ_mask[:,y:y+h,x:x+w] = 1.0
                occ_mask[mask<0.03] = 0.0
            # Misalignment
            else:
                y = random.randint(7, 20)
                x = random.randint(-20, 20)
                if abs(x) < 7:
                    x = 7
                mask_copy = torch.zeros_like(mask)
                if x > 0:
                    mask_copy[:,:-y,x:] = mask[:,y:,:-x]
                else:
                    mask_copy[:,:-y,:x] = mask[:,y:,-x:]

                mask_copy = 1.-mask_copy
                mask_copy[mask<0.03] = 0.0
                erasing = -1 * torch.rand(3,1,1) + 1
                erasing = erasing.expand_as(rotated)

                # Give random noise
                if torch.rand(1) < 0.5:
                    noise = -0.5*torch.rand(3,self.img_size,self.img_size) + 0.5
                    erasing = erasing + noise

                if torch.rand(1) < 0.5:
                    y = random.randint(int(self.img_size*0.3),int(self.img_size*0.75))
                    x = random.randint(int(self.img_size*0.3),int(self.img_size*0.75))
                    mask_copy[:, y:y+15,:] = 0.0
                    mask_copy[:, :, x:x+15] = 0.0
                
                occ_mask[mask_copy==1.0] = 1.0
                mask_copy = mask_copy.repeat(3,1,1)
                rotated[mask_copy==1.0] = erasing[mask_copy==1.0]

            rotated = torch.clamp(rotated, 0.0, 1.0)
        
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[2])
            rotated = torch.flip(rotated, dims=[2])
            guidance = torch.flip(guidance, dims=[2])
            occ_mask = torch.flip(occ_mask, dims=[2])
            
        return img, rotated, guidance, occ_mask
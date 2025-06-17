# signature_dataset.py

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class SignatureDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        
        individuals = os.listdir(root_dir) 
        
        for person in individuals:
            person_dir = os.path.join(root_dir, person)

            if "_forg" in person:
                continue  

            genuine_path = os.path.join(root_dir, person) 
            forged_path = os.path.join(root_dir, person + "_forg")  
            
            if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
                continue  
            
            genuine_signs = sorted(os.listdir(genuine_path))  
            forged_signs = sorted(os.listdir(forged_path))   
            
            if len(genuine_signs) < 2 or len(forged_signs) < 1:
                continue  
            
            # Positive pairs
            for i in range(len(genuine_signs) - 1):
                for j in range(i + 1, len(genuine_signs)):
                    self.pairs.append((os.path.join(genuine_path, genuine_signs[i]),
                                       os.path.join(genuine_path, genuine_signs[j]), 1))

            # Negative pairs
            for i in range(len(genuine_signs)):
                for j in range(len(forged_signs)):
                    self.pairs.append((os.path.join(genuine_path, genuine_signs[i]),
                                       os.path.join(forged_path, forged_signs[j]), 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = Image.open(img1_path).convert("L")  
        img2 = Image.open(img2_path).convert("L")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

import torch.nn as nn
import torch.nn.functional as F
import torch
import os 

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # ENCODER
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),    
            nn.LeakyReLU(0.2)
        )   
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),    
            nn.LeakyReLU(0.2)
        )   
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )   
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),    
            nn.LeakyReLU(0.2)
        )
        
        # DECODER
        #         
        # Deconv1
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.deconv1_norm_act = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5) 
        )
        
        # Deconv2
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2)
        self.deconv2_norm_act = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.5) 
        )

        # Deconv3
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2)
        self.deconv3_norm_act = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.5)  
        )

        # Deconv4
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2)
        self.deconv4_norm_act = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU()
            
        )

        # Deconv5
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2, padding=2)
        self.deconv5_norm_act = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Deconv6 
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.L1Loss()
        
        # Loss tracking
        self.loss_history = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # FORWARD
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, 1, 512, 128)
        Returns:
            mask: (batch, 1, 512, 128)
        """
        # ENCODER
        conv1_out = self.conv1(x)          
        conv2_out = self.conv2(conv1_out)  
        conv3_out = self.conv3(conv2_out)   
        conv4_out = self.conv4(conv3_out)   
        conv5_out = self.conv5(conv4_out)   
        conv6_out = self.conv6(conv5_out)  
        
        # DECODER 
        
        # Deconv1
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_norm_act(deconv1_out) 
        
        # Deconv2
        deconv2_in = torch.cat([deconv1_out, conv5_out], dim=1)  
        deconv2_out = self.deconv2(deconv2_in, output_size=conv4_out.size())
        deconv2_out = self.deconv2_norm_act(deconv2_out)  
        
        # Deconv3
        deconv3_in = torch.cat([deconv2_out, conv4_out], dim=1)  
        deconv3_out = self.deconv3(deconv3_in, output_size=conv3_out.size())
        deconv3_out = self.deconv3_norm_act(deconv3_out)  
        
        # Deconv4
        deconv4_in = torch.cat([deconv3_out, conv3_out], dim=1)  
        deconv4_out = self.deconv4(deconv4_in, output_size=conv2_out.size())
        deconv4_out = self.deconv4_norm_act(deconv4_out) 
        
        # Deconv5
        deconv5_in = torch.cat([deconv4_out, conv2_out], dim=1)  
        deconv5_out = self.deconv5(deconv5_in, output_size=conv1_out.size())
        deconv5_out = self.deconv5_norm_act(deconv5_out)  
        
        # Deconv6
        deconv6_in = torch.cat([deconv5_out, conv1_out], dim=1)  
        deconv6_out = self.deconv6(deconv6_in, output_size=x.size())
        
        # Sigmoid for mask [0, 1]
        mask = torch.sigmoid(deconv6_out)
        
        return mask


    # BACKWARD
    def backward(self, mix, vocal):
        """
        Training step
        
        Args:
            mix: (batch, 1, 512, 128)
            vocal: (batch, 1, 512, 128)
        """
        self.optimizer.zero_grad()        
        mask = self.forward(mix)
        predicted_vocal = mask * mix
        loss = self.criterion(predicted_vocal, vocal)
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()
    
    # SAVE & LOAD
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history': self.loss_history
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            print(f"Loading model from {path}")
            state = torch.load(path)
            self.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            if 'loss_history' in state:
                self.loss_history = state['loss_history']
            print("Model loaded successfully!")
        else:
            print(f"Model file {path} not found. Starting from scratch.")
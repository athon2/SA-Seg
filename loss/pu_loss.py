import torch
import torch.nn as  nn
import torch.nn.functional as F

class LPE_theta1_loss(nn.Module):
    def forward(self, p_y_condi_x, p_y_condi_sx):            
        loss = F.binary_cross_entropy_with_logits(p_y_condi_x, p_y_condi_sx)
        return loss.mean()
    
class LPE_theta2_loss(nn.Module):
    def forward(self, p_s_condi_yx, s, p_y_condi_sx):            
        loss = F.binary_cross_entropy_with_logits(p_s_condi_yx, s, weight=p_y_condi_sx)
        return loss.mean()
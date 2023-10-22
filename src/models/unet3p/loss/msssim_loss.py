import torch
from torch import nn
import torchmetrics

class MSSSIM_Loss(nn.Module):
    def __init__(self, gaussian_kernel=True, kernel_size=11, sigma=1.5, 
                reduction='elementwise_mean', data_range=None, k1=0.01, k2=0.03, 
                betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), normalize='relu'):
        super().__init__()
        self.gaussian_kernel = gaussian_kernel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.betas = betas
        self.normalize = normalize
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input, target):
        ms_ssim = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(gaussian_kernel=self.gaussian_kernel, kernel_size=self.kernel_size,
                sigma=self.sigma, reduction=self.reduction, data_range=self.data_range, k1=self.k1, k2=self.k2, betas=self.betas,
                normalize=self.normalize).to(self.device)
        msssim_value = ms_ssim(input, target)
        loss = 1 - msssim_value
        return loss

if __name__ == "__main__":
    loss_func = MSSSIM_Loss()
    tensor1 = torch.rand((2, 1, 256, 256))
    tensor2 = torch.rand((2, 1, 256, 256))
    loss = loss_func(tensor1, tensor2)
    print(loss)
import numpy as np
import torch

class NoiseScheduler:
    def __init__(self, config):
        self.beta0 = config.beta0
        self.betaT = config.betaT
        self.T = config.T
        self.schedule_type = config.schedule_type
        self.device = config.device
        self.precompute()

    def set_betas_alphas_cumprod(self):
        if self.schedule_type == "linear":
            self.betas = np.linspace(self.beta0, self.betaT, self.T)
            self.alphas = 1 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas)

        elif self.schedule_type == "quadratic":
            self.betas = np.linspace(self.beta0 ** 0.5, self.betaT ** 0.5, self.T) ** 2
            self.alphas = 1. - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas)

        elif self.schedule_type == "sigmoid":
            self.betas = np.linspace(-6, 6, self.T)
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            self.betas = sigmoid(self.betas) * (self.betaT - self.beta0) + self.beta0
            self.alphas = 1. - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas)

        else:
            raise ValueError("Schedule type not recognized")

    def precompute(self):
        self.set_betas_alphas_cumprod()
        
        self.mean_scales = torch.tensor(np.sqrt(self.alphas_cumprod), dtype=torch.float32).to(self.device)
        self.std_scales = torch.tensor(np.sqrt(1 - self.alphas_cumprod), dtype=torch.float32).to(self.device)

        self.inv_sqrt_alphas = torch.tensor(np.sqrt(1 / self.alphas), dtype=torch.float32).to(self.device)
        self.inv_std_scales = torch.tensor(1/np.sqrt(1 - self.alphas_cumprod), dtype=torch.float32).to(self.device)
        self.alphas = torch.tensor(self.alphas, dtype=torch.float32).to(self.device)
        self.sigmas = torch.tensor(np.sqrt(self.betas), dtype=torch.float32).to(self.device)
    
    def __call__(self, x, t, noise):
        mean_scales = self.mean_scales[t]
        std_scales = self.std_scales[t]

        #mean_scales = mean_scales.reshape(-1, 1)
        #std_scales = std_scales.reshape(-1, 1)
        mean_scales = mean_scales.unsqueeze(-1)
        std_scales = std_scales.unsqueeze(-1)

        return mean_scales * x + std_scales * noise
    
    def revert(self, model_output, t, sample):
        inv_sqrt_alpha = self.inv_sqrt_alphas[t]
        inv_std_scale = self.inv_std_scales[t]
        alpha = self.alphas[t]
        sigma = self.sigmas[t].clip(1e-20)

        z = torch.zeros_like(model_output)
        if t > 0:
            z = torch.randn_like(model_output)

        return inv_sqrt_alpha * (sample - (1. - alpha)*inv_std_scale*model_output) + sigma*z

    def __len__(self):
        return self.T
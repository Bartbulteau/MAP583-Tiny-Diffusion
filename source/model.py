import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float, device):
        super().__init__()

        assert size % 2 == 0
        self.size = size
        half_size = size // 2

        vector = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        vector = torch.exp(-vector * torch.arange(half_size))
        vector = vector.unsqueeze(0)
        vector = vector.to(device)

        scale = torch.Tensor([scale]).to(device)

        self.register_buffer('vector', vector)
        self.register_buffer('scale', scale)

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        emb = x.unsqueeze(-1) * self.vector
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    
class Block(nn.Module):
    def __init__(self, size: int, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return x + self.act(self.ff(x))
    

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.time_embedding = SinusoidalEmbedding(config.time_embed_dim, config.time_embed_scale, config.device)
        self.input_embedding_1 = SinusoidalEmbedding(config.input_embed_dim, config.input_embed_scale, config.device)
        self.input_embedding_2 = SinusoidalEmbedding(config.input_embed_dim, config.input_embed_scale, config.device)

        concat_dim = config.time_embed_dim + 2 * config.input_embed_dim

        layers = [nn.Linear(concat_dim, config.hidden_dim), nn.GELU()]

        for _ in range(config.n_layers):
            layers.append(Block(config.hidden_dim, config.dropout))

        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, 2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x1 = self.input_embedding_1(x[:, 0])
        x2 = self.input_embedding_2(x[:, 1])
        t = self.time_embedding(t)

        x = torch.cat((x1, x2, t), dim=-1)

        return self.layers(x)
    
class ModelWithLabels(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.time_embedding = SinusoidalEmbedding(config.time_embed_dim, config.time_embed_scale, config.device)
        self.input_embedding_1 = SinusoidalEmbedding(config.input_embed_dim, config.input_embed_scale, config.device)
        self.input_embedding_2 = SinusoidalEmbedding(config.input_embed_dim, config.input_embed_scale, config.device)

        concat_dim = config.time_embed_dim + 2 * config.input_embed_dim + config.label_dim

        layers = [nn.Linear(concat_dim, config.hidden_dim), nn.GELU()]

        for _ in range(config.n_layers):
            layers.append(Block(config.hidden_dim, config.dropout))

        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, 2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):

        x1 = self.input_embedding_1(x[:, 0])
        x2 = self.input_embedding_2(x[:, 1])
        t = self.time_embedding(t)

        x = torch.cat((x1, x2, t, y), dim=-1)

        return self.layers(x)
    
def train(model, config, noise_scheduler, dataloader):
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    for epoch in range(config.n_epochs):
        epoch_loss = 0
        for x in dataloader:
            x = x.to(config.device)

            t = torch.randint(0, len(noise_scheduler)-1, (config.batch_size,)).to(config.device)
            noise = torch.randn_like(x).to(config.device)
            noisy = noise_scheduler(x, t, noise)
            optimizer.zero_grad()
            loss = F.mse_loss(model(noisy, t), noise)
            epoch_loss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)

        if epoch % (config.n_epochs//10) == 0:
            print(f'Epoch {epoch} loss={epoch_loss:.5f} lr={optimizer.param_groups[0]["lr"]}')
    return losses

def train_with_labels(model, config, noise_scheduler, dataloader):
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    for epoch in range(config.n_epochs):
        epoch_loss = 0
        for x, y in dataloader:
            x = x.to(config.device)
            y = y.to(config.device)

            t = torch.randint(0, len(noise_scheduler), (config.batch_size,)).to(config.device)
            noise = torch.randn_like(x).to(config.device)
            noisy = noise_scheduler(x, t, noise)

            optimizer.zero_grad()
            loss = F.mse_loss(model(noisy, t, y), noise)
            epoch_loss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
        
        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)

        losses.append(epoch_loss)

        if epoch % (config.n_epochs//10) == 0:
            print(f'Epoch {epoch} loss={epoch_loss:.5f} lr={optimizer.param_groups[0]["lr"]}')

    return losses
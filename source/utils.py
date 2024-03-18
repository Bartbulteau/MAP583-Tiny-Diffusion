import torch
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

@torch.no_grad()
def sample(model, config, noise_scheduler):
    model.eval()
    sample = torch.randn(2000, 2).to(config.device)
    frames = [sample.cpu().numpy()]
    ts = np.linspace(0, len(noise_scheduler)-1, len(noise_scheduler))[::-1]
    for t in ts:
        t_ = torch.full((2000,), t).int().to(config.device)
        sample = noise_scheduler.revert(model(sample, t_.float()), t_[0], sample)
        frames.append(sample.cpu().numpy())
    return frames

def make_gif_one(frames, save_path):
    fig, axs = plt.subplots(2, 5, figsize=(15, 5))
    axs = axs.flatten()
    for i in range(10):
        axs[i].set_xlim(-1, 1)
        axs[i].set_ylim(-1, 1)
        axs[i].set_aspect('equal')
        axs[i].set_title(f'Sample {i}')
        axs[i].xaxis.set_tick_params(labelbottom=False)
        axs[i].yaxis.set_tick_params(labelleft=False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    scats = [axs[i].scatter(frames[0][i][:,0], frames[0][i][:,1], s=5, marker='s', alpha=0.2, c='black') for i in range(10)]
    def animate(i):
        i = i*5
        for j in range(10):
            scats[j].set_offsets(frames[j][i])
        return scats
    anim = FuncAnimation(fig, animate, frames=len(frames[0])//5 + 1, interval=100)
    anim.save(save_path, dpi=80, writer='imagemagick')
    plt.close()

@torch.no_grad()
def sample_with_label(model, config, noise_scheduler, n_samples, n):
    model.eval()
    sample = torch.randn(n_samples, 2).to(config.device)
    frames = [sample.cpu().numpy()]
    ts = np.linspace(0, len(noise_scheduler)-1, len(noise_scheduler))[::-1]
    y = torch.zeros((n_samples, config.label_dim)).to(config.device)
    y[:, n] = 1
    y = y.float()
    for t in ts:
        t_ = torch.full((n_samples,), t).int().to(config.device)
        sample = noise_scheduler.revert(model(sample, t_.float(), y), t_[0], sample)
        frames.append(sample.cpu().numpy())
    return frames

def make_gif_nums(frames, save_path):
    fig, axs = plt.subplots(2, 5, figsize=(15, 5))
    axs = axs.flatten()
    for i in range(10):
        axs[i].set_xlim(-1, 1)
        axs[i].set_ylim(-1, 1)
        axs[i].set_aspect('equal')
        axs[i].set_title(f'Sample {i}')
        axs[i].xaxis.set_tick_params(labelbottom=False)
        axs[i].yaxis.set_tick_params(labelleft=False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    scats = [axs[i].scatter(frames[0][i][:,0], frames[0][i][:,1], s=5, marker='s', alpha=0.2, c='black') for i in range(10)]
    def animate(i):
        i = i*5
        for j in range(10):
            scats[j].set_offsets(frames[j][i])
        return scats
    anim = FuncAnimation(fig, animate, frames=len(frames[0])//5 + 1, interval=100)
    anim.save(save_path, dpi=80, writer='imagemagick')
    plt.close()

def make_gif_grecs(frames, save_path):
    fig, axs = plt.subplots(4, 6, figsize=(40, 20))
    axs = axs.flatten()
    for i in range(24):
        axs[i].set_xlim(-1, 1)
        axs[i].set_ylim(-1, 1)
        axs[i].set_aspect('equal')
        axs[i].set_title(f'Sample {i}')
        axs[i].xaxis.set_tick_params(labelbottom=False)
        axs[i].yaxis.set_tick_params(labelleft=False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    scats = [axs[i].scatter(frames[0][i][:,0], frames[0][i][:,1], s=5, marker='s', alpha=0.2, c='black') for i in range(24)]
    def animate(i):
        i = i*5
        for j in range(24):
            scats[j].set_offsets(frames[j][i])
        return scats
    anim = FuncAnimation(fig, animate, frames=len(frames[0])//5 + 1, interval=100)
    anim.save(save_path, dpi=80, writer='imagemagick')
    plt.close()
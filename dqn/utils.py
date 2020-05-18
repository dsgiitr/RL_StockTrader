import os
import torch

PATH = os.path.join('save_model/save.pth')

def save(epoch, model, optimizer, total_rewards, total_losses, epsilon, epochs):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_rewards': total_rewards,
            'total_losses': total_losses,
            'epsilon': epsilon,
            'epochs': epochs
            }, PATH)

def load(model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epsilon = checkpoint['epsilon']
    total_rewards = checkpoint['total_rewards']
    total_losses = checkpoint['total_losses']
    epochs = checkpoint['epochs']
    return epoch, model, optimizer, epsilon, total_losses, total_rewards, epochs



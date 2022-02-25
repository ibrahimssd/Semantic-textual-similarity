import torch
import matplotlib.pyplot as plt

def similarity_score(input1, input2):
    # Get similarity predictions:
    dif = input1.squeeze() - input2.squeeze()

    norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
    y_hat = torch.exp(-norm)
    y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    return y_hat


# plot losses and accuracy progress

def plot_progress(stats):
    train_losses= stats['train_loss']
    val_losses=   stats['val_loss']
    epochs=   stats['epochs']
    
    
    fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=100)

    # plot subplot 1
    axes[0].plot(epochs, train_losses, color="green", label="train loss")
    axes[0].plot(epochs, val_losses, color="red", label="val loss")
    axes[0].legend(loc=2); # upper left corner
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Plot training loss and validation loss')

    # plot subplot 2
    axes[1].plot(epochs, train_losses, color="green", label="train loss")
    axes[1].plot(epochs, val_losses, color="red", label="val loss")
    axes[1].legend(loc=2); # upper left corner
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('loss')
    axes[1].set_title('Plot training loss and validation loss')
   
    fig.tight_layout()

    plt.show()
import torch
import matplotlib.pyplot as plt


def predict_compare(model, device, dataloader, num_samples=3):
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            if count >= num_samples: break
            imgs = imgs.to(device)
            out = model(imgs)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()
            imgs_np = imgs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            pred_np = pred.cpu().numpy()
            for i in range(min(num_samples - count, imgs.size(0))):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(imgs_np[i][0], cmap='gray')
                axes[0].axis('off')
                axes[0].set_title('Input')

                axes[1].imshow(masks_np[i][0], cmap='gray')
                axes[1].axis('off')
                axes[1].set_title('Ground Truth')

                axes[2].imshow(pred_np[i][0], cmap='gray')
                axes[2].axis('off')
                axes[2].set_title('Prediction')

                plt.show()
                plt.close(fig)
                count += 1


def plot_history_loss(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], '.', label='Train Loss')
    plt.plot(epochs, history['val_loss'], '.', label='Val Loss')
    plt.legend()
    plt.show()

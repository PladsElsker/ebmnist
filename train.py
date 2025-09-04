import mlflow
import torch
from random import randint, random
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import math

from custom_blocks import EBMSamplerV2, BWImageScalarEnergy, LinearAutoEncoderV2


def main():
    train_v1()


def train_v1():
    device = 'cuda:4'
    dataset = datasets.MNIST(root='ebmnist/mnist', download=True, transform=to_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    batch_size = 32
    grad_acc = 4

    grad_step = 0

    train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_dataset = DataLoader(validation_dataset, batch_size=batch_size, num_workers=8)

    if False:
        datapoint = next(iter(train_dataset))
        train_dataset = [datapoint for _ in range(len(train_dataset))]

        datapoint = next(iter(train_dataset))
        validation_dataset = [datapoint for _ in range(len(train_dataset))][:len(validation_dataset)]

    autoencoder = LinearAutoEncoderV2(patch_size=4, embed_dim=192).to(device)
    autoencoder.requires_grad_(False)
    autoencoder.eval()

    pretrained_model_path = None
    model = BWImageScalarEnergy(autoencoder=autoencoder, embed_dim=192, num_blocks=6, num_heads=4, dropout=0).to(device)
    
    model_filename = 'ebmnist/models/BWImageScalarEnergy-Test3-{epoch}.pth'

    model.apply(init_weights)
    # model.apply(init_mha_value_zero)

    model = EBMSamplerV2(model)

    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path))

    lr = 1e-4
    epochs = 20
    epoch_start = 1
    criterions = {
        'MSE': nn.MSELoss(),
    }
    criterion = lambda y, y_hat: torch.mean(torch.stack([crit(y, y_hat) for crit in criterions.values()]))

    previous_optimizer_path = None
    optimizer = torch.optim.AdamW(lr=lr, params=model.parameters())
    if previous_optimizer_path is not None:
        optimizer.load_state_dict(torch.load(previous_optimizer_path))

    UPDATE_VISUALISATION_EVERY = 8
    visualisation_counter = 0

    def step_size(mi=0.08, ma=0.12):
        return random() * (ma - mi) + mi

    def steps():
        return randint(48, 52)

    mlflow.set_tracking_uri('http://192.168.0.210:7961')
    mlflow.set_experiment("EBMNIST")

    train_losses_for_epochs = []
    validation_losses_for_epochs = []

    with mlflow.start_run():
        mlflow.log_artifact('ebmnist/train.py', artifact_path='source_code')
        mlflow.log_artifact('ebmnist/custom_blocks.py', artifact_path='source_code')
        mlflow.log_params({
            'learning_rate': lr, 
            'batch_size': batch_size, 
            'grad_acc': grad_acc, 
            'epoch_start': epoch_start, 
            'epoch_end': epochs, 
            'base_model': pretrained_model_path,
        })

        for epoch_id in range(epoch_start, epochs + 1):
            print('Epoch:', epoch_id)

            train_losses = []
            validation_losses = []

            if Path(model_filename.format(epoch=str(epoch_id))).exists():
                raise RuntimeError(f'Model name {model_filename.format(epoch=str(epoch_id))} already exists. Change it.')

            with tqdm(total=len(train_dataset), desc="Train") as pbar:
                model.train()
                autoencoder.eval()

                for batch_id, (y, x) in enumerate(train_dataset):
                    batch_size = x.shape[0]
                    x = x.to(torch.float).to(device).unsqueeze(1)
                    y = y.to(device)

                    sampled_steps = steps()
                    sampled_step_size = step_size()
                    noise = torch.randn_like(y)
                    alpha = torch.rand(batch_size).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
                    y_hat = y * (1-alpha) + alpha * noise
                    y_hat_copy = y_hat.detach().cpu()

                    y_hat, e_traj = model(x, y_hat, sampled_steps, sampled_step_size, noise=math.sqrt(2*sampled_step_size))

                    loss = criterion(y_hat, y) / grad_acc
                    loss.backward()

                    train_losses.append(loss.detach().item())
                    mlflow.log_metric('Train Loss', loss, step=(epoch_id-1) * len(train_dataset) + batch_id)

                    grad_step += 1
                    if grad_step >= grad_acc:
                        grad_step = 0
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        params_norm = torch.nn.utils.get_total_norm(model.model.parameters())
                        mlflow.log_metric('Params L2 Norm', params_norm, step=(epoch_id-1) * len(train_dataset) + batch_id)
                        optimizer.step()
                        optimizer.zero_grad()

                    visualisation_counter += 1
                    if visualisation_counter >= UPDATE_VISUALISATION_EVERY:
                        visualisation_counter = 0
                        update_visualisation(x[0], y[0], y_hat_copy[0], y_hat[0], alpha[0].item(), e_traj[0])

                    pbar.set_description(f'Train - Loss: {sum(train_losses) / len(train_losses)}')
                    pbar.update(1)

            model.eval()

            for batch_id, (y, x) in enumerate(tqdm(validation_dataset, desc='Validation')):
                batch_size = x.shape[0]
                x = x.to(torch.float).to(device).unsqueeze(1)
                y = y.to(device)
                noise = torch.randn_like(y)
                alpha = torch.rand(batch_size).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
                y_hat = y.detach() * (1-alpha) + alpha * noise
                y_hat_copy = y_hat.detach().cpu()

                sampled_steps = steps() * 2
                sampled_step_size = step_size()

                y_hat, e_traj = model(x, y_hat, sampled_steps, sampled_step_size, create_graph=False)

                loss = criterion(y_hat, y) / grad_acc

                validation_losses.append(loss.detach().item())
                mlflow.log_metric('Validation Loss', loss, step=(epoch_id-1) * len(validation_dataset) + batch_id)

                visualisation_counter += 1
                if visualisation_counter >= UPDATE_VISUALISATION_EVERY:
                    visualisation_counter = 0
                    update_visualisation(x[0], y[0], y_hat_copy[0], y_hat[0], alpha[0].item(), e_traj[0])

            train_losses_for_epochs.append(sum(train_losses) / len(train_losses))
            validation_losses_for_epochs.append(sum(validation_losses) / len(validation_losses))

            print('Train loss:', train_losses_for_epochs[-1])
            print('Validation loss:', validation_losses_for_epochs[-1])
            
            mlflow.log_metric(f"Train Loss Epoch", train_losses_for_epochs[-1], step=epoch_id)
            mlflow.log_metric(f"Validation Loss Epoch", validation_losses_for_epochs[-1], step=epoch_id)

            epochs_count = range(epoch_start, epoch_id+1)

            _, axes = plt.subplots(1, 1, figsize=(12, 8))

            style = 'o' if len(epochs_count) == 1 else '-'

            axes.plot(epochs_count, train_losses_for_epochs, style, label="Train Loss")
            axes.plot(epochs_count, validation_losses_for_epochs, style, label="Validation Loss")

            axes.set_title("Loss per Epoch")
            axes.set_xlabel("Epoch")
            axes.set_ylabel("Loss")
            axes.legend()
            axes.grid()

            plt.tight_layout()
            plt.savefig('ebmnist/loss_curves.png')
            mlflow.log_artifact('ebmnist/loss_curves.png', 'plots')

            torch.save(model.state_dict(), model_filename.format(epoch=str(epoch_id)))
            torch.save(model.state_dict(), model_filename.format(epoch='last'))
            print(f'Saved model {model_filename.format(epoch=str(epoch_id))}')
            torch.save(optimizer.state_dict(), optimizer_path := 'ebmnist/models/Adam-state.pth')
            print(f'Saved optimizer state {optimizer_path}')
            print()


def update_visualisation(x, y, noised_sample, predicted, alpha, e_traj):
    with torch.no_grad():
        x = int(x.clone().detach().cpu())
        y = y.clone().detach().cpu().permute(1, 2, 0)
        noised_sample = noised_sample.clone().detach().cpu().permute(1, 2, 0)
        predicted = predicted.clone().detach().cpu().permute(1, 2, 0)
        e_traj = [e.item() for e in e_traj.detach().cpu()]

        fig = plt.figure(figsize=(9, 6))
        fig.suptitle(f"Digit: {x}, Alpha: {alpha}")

        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(noised_sample.clip(0, 1))
        ax0.set_title("Noised Sample")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(predicted.clip(0, 1))
        ax1.set_title("Predicted")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(y.clip(0, 1))
        ax2.set_title("Ground Truth")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(range(len(e_traj)), e_traj)
        ax3.set_title("Energy Trajectory")
        ax3.set_xlabel("Langevin Step")
        ax3.set_ylabel("Energy")
        ax3.grid(True)
        ax3.set_ylim(bottom=0)

        plt.tight_layout()

        plt.savefig('ebmnist/predicions.png')
        mlflow.log_artifact('ebmnist/predicions.png', 'plots')
        plt.close()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        if not any(p.requires_grad for p in m.parameters()):
            return

        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def init_mha_value_zero(module):
    if isinstance(module, nn.MultiheadAttention):
        embed_dim = module.embed_dim
        start = 2 * embed_dim
        end = 3 * embed_dim
        init.zeros_(module.in_proj_weight[start:end, :])
        if module.in_proj_bias is not None:
            init.zeros_(module.in_proj_bias[start:end])


if __name__ == '__main__':
    main()

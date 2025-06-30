import torch, os, wandb, argparse, yaml
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from Vanilla_Backpropagation.bp_network import BPNetwork
from datetime import datetime
from tqdm import tqdm
from multiprocessing import freeze_support

def evaluate(model, loader, device):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x = torch.flatten(x, start_dim=1).to(device)
            y = y.to(device)

            logits = model(x)
            loss_sum += ce_loss(logits, y).item() * x.size(0)

            pred_top1 = logits.argmax(dim=1)
            correct += (pred_top1 == y).sum().item()

            total += y.size(0)

    return {
        "val_loss": loss_sum / total,
        "val_acc":  correct / total,
    }
    
def main():
    cfg = yaml.safe_load(open('configs/bp_configs.yaml'))

    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--bp_lr', type=float)
    p.add_argument('--num_workers', type=int)
    p.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'])
    args = p.parse_args()

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.bp_lr is not None:
        cfg["bp_lr"] = args.bp_lr
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    seed = 42

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device used: " + device)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Run naming for BP training
    now = datetime.now()
    base_name = f"bp_{cfg['dataset']}_bs{cfg['batch_size']}_{now.day}_{now.strftime('%b').lower()}_{now.year}"
    run_number = 1
    while True:
        run_name = f"{base_name}_{run_number}"
        checkpoint_path = os.path.join(cfg["out_dir"], f"run_{run_name}")
        if not os.path.exists(checkpoint_path):
            break
        run_number += 1
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb.login()
    wandb.init(entity="mathys-projects", project=f"comp-490-concordia", name=run_name, config=cfg)

    data_dir = "data/"
    os.makedirs(data_dir, exist_ok=True)

    if cfg["dataset"] == 'mnist':
        train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    elif cfg["dataset"] == 'cifar10':
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    elif cfg["dataset"] == 'cifar100':
        train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        cfg["output_dim"] = 100

    VAL_FRAC = 0.1 # 10 % hold-out
    train_size = int((1-VAL_FRAC) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device=="cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device=="cuda")
    )

    # BP model initialization
    bp_model = BPNetwork(
        input_dim=cfg[cfg["dataset"]]["input_dim"],
        hidden_dims=cfg[cfg["dataset"]]["hidden_dims"],
        output_dim=cfg["output_dim"],
    )
    bp_model.to(device)
    wandb.watch(bp_model, log="all", log_freq=cfg["log_interval"], log_graph=True, criterion=None)

    bp_optimizer = torch.optim.RMSprop(bp_model.parameters(), lr=float(cfg["bp_lr"]))
    bp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bp_optimizer, cfg["epochs"])
    bp_loss_fn = nn.CrossEntropyLoss()

    # BP training loop
    for epoch in range(cfg["epochs"]):
        bp_model.train()
        
        epoch_bp_loss = 0.0
        epoch_bp_correct = 0
        epoch_total_samples = 0

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")):
            x = torch.flatten(x, start_dim=1).to(device)
            y = y.to(device)
            
            bp_optimizer.zero_grad()
            bp_logits = bp_model(x)
            bp_loss_val = bp_loss_fn(bp_logits, y)
            bp_loss_val.backward()
            bp_optimizer.step()
            
            epoch_bp_loss += bp_loss_val.item()
            bp_preds = torch.argmax(bp_logits, dim=1)
            bp_batch_correct = (bp_preds == y).sum().item()
            epoch_bp_correct += bp_batch_correct
            epoch_total_samples += x.size(0)
            
            if batch_idx % cfg["log_interval"] == 0:
                wandb.log({
                    "bp_loss_batch": bp_loss_val.item(),
                    "bp_train_batch_correct_count": bp_batch_correct,
                    "bp_train_batch_acc": bp_batch_correct / x.size(0),
                })
                
        epoch_bp_loss /= len(train_loader)
        epoch_bp_acc = epoch_bp_correct / epoch_total_samples

        print(f"Epoch {epoch}/{cfg['epochs']}: BP Loss={epoch_bp_loss:.4f}, TrainAcc={epoch_bp_acc*100:.2f}%")
        
        log_dict = {
            "epoch": epoch,
            "epoch_loss": epoch_bp_loss,
            "train_epoch_acc": epoch_bp_acc,
        }

        run_eval = ((epoch + 1) % 5 == 0) or (epoch == cfg["epochs"] - 1)
        if run_eval:
            metrics = evaluate(bp_model, val_loader, device)
            print(f" ValLoss={metrics['val_loss']:.4g} ValAcc={100*metrics['val_acc']:.2f}%")
            log_dict.update({
                "val_loss": metrics['val_loss'],
                "val_acc": metrics['val_acc'],
            })

        wandb.log(log_dict)
        
        # Save BP checkpoint every checkpoint_interval epochs or on the final epoch
        if (epoch + 1) % cfg["checkpoint_interval"] == 0 or epoch == cfg["epochs"] - 1:
            bp_ckpt_path = os.path.join(checkpoint_path, f"bp_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": bp_model.state_dict(),
                "optimizer_state_dict": bp_optimizer.state_dict(),
                "cfg": cfg
            }, bp_ckpt_path)
            print(f"BP Checkpoint saved: {bp_ckpt_path}")
            
        bp_sched.step()

    wandb.finish()

if __name__ == '__main__':
    freeze_support()
    main()
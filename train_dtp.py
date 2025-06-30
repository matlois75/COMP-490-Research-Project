import torch, os, os.path, wandb, argparse, yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from Difference_Target_Propagation.dtp_network import DTPNetwork
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

            logits, _ = model(x)
            loss_sum += ce_loss(logits, y).item() * x.size(0)

            pred_top1 = logits.argmax(dim=1)
            correct += (pred_top1 == y).sum().item()

            total += y.size(0)

    return {
        "val_loss": loss_sum / total,
        "val_acc":  correct / total,
    }
        
def main():
    cfg = yaml.safe_load(open('configs/dtp_configs.yaml'))

    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--forward_lr', type=float)
    p.add_argument('--inverse_lr', type=float)
    p.add_argument('--num_workers', type=int)
    p.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'])
    args = p.parse_args()

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.forward_lr is not None:
        cfg["forward_lr"] = args.forward_lr
    if args.inverse_lr is not None:
        cfg["inverse_lr"] = args.inverse_lr
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

    # meaningful run naming
    now = datetime.now()
    base_name = f"dtp_{cfg['dataset']}_{now.day}_{now.strftime('%b').lower()}_{now.year}"
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
        train_dataset = MNIST(
            root=data_dir, train=True, download=True,
            transform=transforms.ToTensor()
        )
    elif cfg["dataset"] == 'cifar10':
        train_dataset = CIFAR10(
            root=data_dir, train=True, download=True,
            transform=transforms.ToTensor()
        )
    elif cfg["dataset"] == 'cifar100':
        train_dataset = CIFAR100(
            root=data_dir, train=True, download=True,
            transform=transforms.ToTensor()
        )
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

    # initialize model and optimizer
    dtp_model = DTPNetwork(
        input_dim=cfg[cfg["dataset"]]["input_dim"],
        hidden_dims=cfg[cfg["dataset"]]["hidden_dims"],
        output_dim=cfg["output_dim"],
        eta_hat=cfg["eta_hat"],
        sigma=cfg["sigma"],
    )
    dtp_model.to(device)
    wandb.watch(dtp_model, log="all", log_freq=cfg["log_interval"], log_graph=True, criterion=None)

    forward_params = []
    inverse_params = []
    for name, param in dtp_model.named_parameters():
        if 'f_layers' in name:
            forward_params.append(param)
        elif 'g_layers' in name:
            inverse_params.append(param)
        else:
            forward_params.append(param) # maybe add to a third group instead?

    f_optimizer = torch.optim.RMSprop(forward_params, lr=float(cfg["forward_lr"]), weight_decay=float(cfg["forward_lr_decay"]))
    g_optimizer = torch.optim.RMSprop(inverse_params, lr=float(cfg["inverse_lr"]))
    
    f_sched = torch.optim.lr_scheduler.CosineAnnealingLR(f_optimizer, cfg["epochs"])
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, cfg["epochs"])

    # training loop
    for epoch in range(cfg["epochs"]):
        dtp_model.train()
        
        epoch_dtp_loss_top = 0.0
        epoch_dtp_f_loss = 0.0
        epoch_dtp_g_loss = 0.0
        epoch_dtp_correct = 0
        epoch_total_samples = 0

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")):
            x = torch.flatten(x, start_dim=1).to(device)
            y = y.to(device)
            
            # feedback‑only warm‑up + K inverse updates per batch
            for _ in range(cfg["k_g_updates"]):
                _, _, g_loss_val, _ = dtp_model.step(x, y, g_optimizer, update_forward=False)
                
            dtp_loss_top_val, dtp_f_loss_val, dtp_g_loss_val, dtp_batch_correct = dtp_model.step(x, y, (f_optimizer, g_optimizer), update_forward=True)
            
            epoch_dtp_loss_top += dtp_loss_top_val
            epoch_dtp_f_loss += dtp_f_loss_val
            epoch_dtp_g_loss += dtp_g_loss_val
            epoch_dtp_correct += dtp_batch_correct
            epoch_total_samples += x.size(0)
            
            if batch_idx % cfg["log_interval"] == 0:
                wandb.log({
                    "loss_top_batch": dtp_loss_top_val,
                    "f_loss_batch": dtp_f_loss_val,
                    "g_loss_batch": dtp_g_loss_val,
                    "train_batch_correct_count": dtp_batch_correct,
                    "train_batch_acc": dtp_batch_correct / x.size(0),
                })
                
        epoch_dtp_loss_top /= len(train_loader)
        epoch_dtp_f_loss /= len(train_loader)
        epoch_dtp_g_loss /= len(train_loader)
        epoch_dtp_acc = epoch_dtp_correct / epoch_total_samples

        print(f"Epoch {epoch}/{cfg['epochs']}:")
        print(f"DTP: TopLoss={epoch_dtp_loss_top:.4g}, F-Loss={epoch_dtp_f_loss:.4g}, G-Loss={epoch_dtp_g_loss:.4g}, TrainAcc={epoch_dtp_acc*100:.2f}%")
        
        run_eval = ((epoch + 1) % 5 == 0) or (epoch == cfg["epochs"] - 1)
        if run_eval:
            metrics = evaluate(dtp_model, val_loader, device)
            print(f"  ValLoss={metrics['val_loss']:.4g}  ValAcc={100*metrics['val_acc']:.2f}%")
            wandb.log({
                "epoch": epoch,
                "epoch_loss_top": epoch_dtp_loss_top,
                "epoch_f_loss":   epoch_dtp_f_loss,
                "epoch_g_loss":   epoch_dtp_g_loss,
                "train_epoch_acc": epoch_dtp_acc,
                **metrics
            })
        else:
            wandb.log({
                "epoch": epoch,
                "epoch_loss_top": epoch_dtp_loss_top,
                "epoch_f_loss":   epoch_dtp_f_loss,
                "epoch_g_loss":   epoch_dtp_g_loss,
                "train_epoch_acc": epoch_dtp_acc,
            })
        
        # save checkpoint every N epochs or on the last epoch
        if (epoch + 1) % cfg["checkpoint_interval"] == 0 or epoch == cfg["epochs"] - 1:
            dtp_ckpt_path = os.path.join(checkpoint_path, f"dtp_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": dtp_model.state_dict(),
                "f_optimizer_state_dict": f_optimizer.state_dict(),
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                "cfg": cfg
            }, dtp_ckpt_path)
            print(f"DTP Checkpoint saved: {dtp_ckpt_path}")
            
        f_sched.step()
        g_sched.step()

    wandb.finish()
    
if __name__ == '__main__':
    freeze_support()
    main()
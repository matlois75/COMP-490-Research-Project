import torch, os, os.path, wandb, argparse, yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from Difference_Target_Propagation.dtp_network import DTPNetwork
from datetime import datetime
from tqdm import tqdm

HYPERPARAMS = yaml.safe_load(open('configs/dtp_configs.yaml'))

p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int)
p.add_argument('--batch_size', type=int)
p.add_argument('--forward_lr', type=float)
p.add_argument('--inverse_lr', type=float)
p.add_argument('--num_workers', type=int)
p.add_argument('--dataset', choices=['mnist', 'cifar10'])
args = p.parse_args()

if args.epochs is not None:
    HYPERPARAMS["epochs"] = args.epochs
if args.batch_size is not None:
    HYPERPARAMS["batch_size"] = args.batch_size
if args.forward_lr is not None:
    HYPERPARAMS["forward_lr"] = args.forward_lr
if args.inverse_lr is not None:
    HYPERPARAMS["inverse_lr"] = args.inverse_lr
if args.num_workers is not None:
    HYPERPARAMS["num_workers"] = args.num_workers
if args.dataset is not None:
    HYPERPARAMS["dataset"] = args.dataset

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
base_name = f"dtp_{HYPERPARAMS['dataset']}_bs{HYPERPARAMS['batch_size']}_{now.day}_{now.strftime('%b').lower()}_{now.year}"
run_number = 1
while True:
    run_name = f"{base_name}_{run_number}"
    checkpoint_path = os.path.join(HYPERPARAMS["out_dir"], f"run_{run_name}")
    if not os.path.exists(checkpoint_path):
        break
    run_number += 1
os.makedirs(HYPERPARAMS["out_dir"], exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

wandb.login()
wandb.init(entity="comp-490-concordia", project=f"dtp_{HYPERPARAMS['dataset']}", name=run_name, config=HYPERPARAMS)

data_dir = "data/"
os.makedirs(data_dir, exist_ok=True)

if HYPERPARAMS["dataset"] == 'mnist':
    train_dataset = MNIST(
        root=data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
elif HYPERPARAMS["dataset"] == 'cifar10':
    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # keeps activations in the linear regions of tanh
        ])
    )

train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True, num_workers=HYPERPARAMS["num_workers"], pin_memory=(device=="cuda"))

# initialize model and optimizer
dtp_model = DTPNetwork(
    input_dim=HYPERPARAMS[HYPERPARAMS["dataset"]]["input_dim"],
    hidden_dims=HYPERPARAMS[HYPERPARAMS["dataset"]]["hidden_dims"],
    output_dim=HYPERPARAMS["output_dim"],
    eta_hat=HYPERPARAMS["eta_hat"],
    sigma=HYPERPARAMS["sigma"]
)
dtp_model.to(device)
wandb.watch(dtp_model, log="all", log_freq=HYPERPARAMS["log_interval"], log_graph=True, criterion=None)

forward_params = []
inverse_params = []
for name, param in dtp_model.named_parameters():
    if 'f_layers' in name:
        forward_params.append(param)
    elif 'g_layers' in name:
        inverse_params.append(param)
    else:
        forward_params.append(param) # maybe add to a third group instead?

f_optimizer = torch.optim.RMSprop(forward_params, lr=float(HYPERPARAMS["forward_lr"]))
g_optimizer = torch.optim.RMSprop(inverse_params, lr=float(HYPERPARAMS["inverse_lr"]))

# training loop
for epoch in range(HYPERPARAMS["epochs"]):
    dtp_model.train()
    
    epoch_dtp_loss_top = 0.0
    epoch_dtp_f_loss = 0.0
    epoch_dtp_g_loss = 0.0
    epoch_dtp_correct = 0
    epoch_total_samples = 0

    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{HYPERPARAMS['epochs']}")):
        x = x = torch.flatten(x, start_dim=1).to(device)
        y = y.to(device)
        
        # feedback‑only warm‑up + K inverse updates per batch
        for _ in range(HYPERPARAMS["k_g_updates"]):
            _, _, g_loss_val, _ = dtp_model.step(x, y, g_optimizer, update_forward=False)
            
        dtp_loss_top_val, dtp_f_loss_val, dtp_g_loss_val, dtp_batch_correct = dtp_model.step(x, y, (f_optimizer, g_optimizer), update_forward=True)
        
        epoch_dtp_loss_top += dtp_loss_top_val
        epoch_dtp_f_loss += dtp_f_loss_val
        epoch_dtp_g_loss += dtp_g_loss_val
        epoch_dtp_correct += dtp_batch_correct
        epoch_total_samples += x.size(0)
        
        if batch_idx % HYPERPARAMS["log_interval"] == 0:
            wandb.log({
                "dtp_loss_top_batch": dtp_loss_top_val,
                "dtp_f_loss_batch": dtp_f_loss_val,
                "dtp_g_loss_batch": dtp_g_loss_val,
                "dtp_train_batch_correct_count": dtp_batch_correct,
                "dtp_train_batch_acc": dtp_batch_correct / x.size(0),
            })
            
    epoch_dtp_loss_top /= len(train_loader)
    epoch_dtp_f_loss /= len(train_loader)
    epoch_dtp_g_loss /= len(train_loader)
    epoch_dtp_acc = epoch_dtp_correct / epoch_total_samples

    print(f"Epoch {epoch}/{HYPERPARAMS['epochs']}:")
    print(f"DTP: TopLoss={epoch_dtp_loss_top:.4g}, F-Loss={epoch_dtp_f_loss:.4g}, G-Loss={epoch_dtp_g_loss:.4g}, TrainAcc={epoch_dtp_acc*100:.2f}%")
    
    wandb.log({
        "epoch": epoch,
        "dtp_epoch_loss_top": epoch_dtp_loss_top,
        "dtp_epoch_f_loss": epoch_dtp_f_loss,
        "dtp_epoch_g_loss": epoch_dtp_g_loss,
        "dtp_train_epoch_acc": epoch_dtp_acc,
    })
    
    # save checkpoint every N epochs or on the last epoch
    if (epoch + 1) % HYPERPARAMS["checkpoint_interval"] == 0 or epoch == HYPERPARAMS["epochs"] - 1:
        dtp_ckpt_path = os.path.join(checkpoint_path, f"dtp_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": dtp_model.state_dict(),
            "f_optimizer_state_dict": f_optimizer.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "cfg": HYPERPARAMS
        }, dtp_ckpt_path)
        print(f"DTP Checkpoint saved: {dtp_ckpt_path}")

wandb.finish()
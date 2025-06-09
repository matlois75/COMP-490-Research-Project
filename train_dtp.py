import torch, os, os.path, wandb, argparse, yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from Difference_Target_Propagation.dtp_network import DTPNetwork
from datetime import datetime

HYPERPARAMS = yaml.safe_load(open('configs/dtp_mnist.yaml'))

p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int)
p.add_argument('--batch_size', type=int)
p.add_argument('--lr', type=float)
p.add_argument('--num_workers', type=int)
args = p.parse_args()

if args.epochs is not None:
    HYPERPARAMS["epochs"] = args.epochs
if args.batch_size is not None:
    HYPERPARAMS["batch_size"] = args.batch_size
if args.lr is not None:
    HYPERPARAMS["learning_rate"] = args.lr
if args.num_workers is not None:
    HYPERPARAMS["num_workers"] = args.num_workers

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
base_name = f"dtp_lr{HYPERPARAMS['learning_rate']}_bs{HYPERPARAMS['batch_size']}_{now.day}_{now.strftime('%b').lower()}_{now.year}"
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
wandb.init(entity="comp-490-concordia", project="dtp_mnist", name=run_name, config=HYPERPARAMS)

data_dir = "data/"
os.makedirs(data_dir, exist_ok=True)

train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

# batch and shuffle training data; no shuffling for test set
train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True, num_workers=HYPERPARAMS["num_workers"], pin_memory=(device=="cuda"))

model = DTPNetwork(input_dim=HYPERPARAMS["input_dim"], hidden_dims=HYPERPARAMS["hidden_dims"], output_dim=HYPERPARAMS["output_dim"], eta_hat=HYPERPARAMS["eta_hat"], sigma=HYPERPARAMS["sigma"])
model.to(device)
wandb.watch(model, log="all", log_freq=HYPERPARAMS["log_interval"])

forward_params = []
inverse_params = []
for name, param in model.named_parameters():
    if 'f_layers' in name:
        forward_params.append(param)
    elif 'g_layers' in name:
        inverse_params.append(param)
    else:
        forward_params.append(param) # maybe add to a third group instead

optimizer = torch.optim.RMSprop([
    {'params': forward_params, 'lr': HYPERPARAMS["forward_lr"]},
    {'params': inverse_params, 'lr': HYPERPARAMS["inverse_lr"]},
])
loss_fn = nn.CrossEntropyLoss() # use in eval only, training uses DTP's internal losses

initial_sigma_val = HYPERPARAMS["initial_sigma"]
final_sigma_val = HYPERPARAMS["final_sigma"]
sigma_decay_epochs_val = HYPERPARAMS["epochs"]/2

# training loop
for epoch in range(HYPERPARAMS["epochs"]):
    model.train()
    
    # calculate current sigma for decay
    if epoch < sigma_decay_epochs_val:
        current_sigma = initial_sigma_val - (initial_sigma_val - final_sigma_val) * (epoch / sigma_decay_epochs_val)
    else:
        current_sigma = final_sigma_val
    model.sigma = current_sigma

    epoch_loss_top = 0.0
    epoch_f_loss = 0.0
    epoch_g_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = torch.flatten(x, start_dim=1).to(device) # flatten from (batch_size, 1, 28, 28) to (batch_size, 784)
        y = y.to(device)
        
        loss_top_val, f_loss_val, g_loss_val, batch_correct = model.step(x=x, labels=y, optimizer=optimizer)
        
        epoch_loss_top += loss_top_val
        epoch_f_loss += f_loss_val
        epoch_g_loss += g_loss_val
        epoch_correct += batch_correct
        epoch_total += x.size(0)
        
        if batch_idx % HYPERPARAMS["log_interval"] == 0:
            wandb.log({
                "loss_top_batch": loss_top_val,
                "f_loss_batch": f_loss_val,
                "g_loss_batch": g_loss_val,
                "train_batch_correct": batch_correct,
            })
            
    epoch_loss_top /= len(train_loader)
    epoch_f_loss /= len(train_loader)
    epoch_g_loss /= len(train_loader)
    epoch_acc = epoch_correct / epoch_total

    print(f"Epoch {epoch}/{HYPERPARAMS['epochs']}: "
      f"TopLoss={epoch_loss_top:.4f}, F-Loss={epoch_f_loss:.4f}, G-Loss={epoch_g_loss:.4f}, "
      f"TrainAcc={epoch_acc*100:.2f}%")
    wandb.log({
        "epoch_loss_top": epoch_loss_top,
        "epoch_f_loss": epoch_f_loss,
        "epoch_g_loss": epoch_g_loss,
        "train_epoch_correct": epoch_acc,
        "epoch": epoch,
        "current_sigma": current_sigma,
    })
    
    # save checkpoint every N epochs or on the last epoch
    if (epoch + 1) % HYPERPARAMS["checkpoint_interval"] == 0 or epoch == HYPERPARAMS["epochs"] - 1:
        ckpt_path = os.path.join(checkpoint_path, f"dtp_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": HYPERPARAMS
        }, ckpt_path)
        # wandb.save(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

wandb.finish()
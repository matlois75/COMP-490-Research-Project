import torch, os, os.path, wandb, argparse, yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from Difference_Target_Propagation.dtp_network import DTPNetwork
from Vanilla_Backpropagation.bp_network import BPNetwork
from datetime import datetime

HYPERPARAMS = yaml.safe_load(open('configs/dtp_mnist.yaml'))

p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int)
p.add_argument('--batch_size', type=int)
p.add_argument('--forward_lr', type=float)
p.add_argument('--inverse_lr', type=float)
p.add_argument('--bp_lr', type=float)
p.add_argument('--num_workers', type=int)
args = p.parse_args()

if args.epochs is not None:
    HYPERPARAMS["epochs"] = args.epochs
if args.batch_size is not None:
    HYPERPARAMS["batch_size"] = args.batch_size
if args.forward_lr is not None:
    HYPERPARAMS["forward_lr"] = args.forward_lr
if args.inverse_lr is not None:
    HYPERPARAMS["inverse_lr"] = args.inverse_lr
if args.bp_lr is not None:
    HYPERPARAMS["bp_lr"] = args.bp_lr
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
base_name = f"dtp_vs_bp_bs{HYPERPARAMS['batch_size']}_{now.day}_{now.strftime('%b').lower()}_{now.year}"
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

dtp_model = DTPNetwork(input_dim=HYPERPARAMS["input_dim"], hidden_dims=HYPERPARAMS["hidden_dims"], output_dim=HYPERPARAMS["output_dim"], eta_hat=HYPERPARAMS["eta_hat"], sigma=HYPERPARAMS["sigma"])
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

dtp_optimizer = torch.optim.RMSprop([
    {'params': forward_params, 'lr': float(HYPERPARAMS["forward_lr"])},
    {'params': inverse_params, 'lr': float(HYPERPARAMS["inverse_lr"])},
])
loss_fn = nn.CrossEntropyLoss() # use in eval only, training uses DTP's internal losses

# BP model
bp_model = BPNetwork(input_dim=HYPERPARAMS["input_dim"], hidden_dims=HYPERPARAMS["hidden_dims"], output_dim=HYPERPARAMS["output_dim"])
bp_model.to(device)
wandb.watch(bp_model, log="all", log_freq=HYPERPARAMS["log_interval"], log_graph=True, criterion=nn.CrossEntropyLoss())

bp_optimizer = torch.optim.RMSprop(bp_model.parameters(), lr=float(HYPERPARAMS["bp_lr"]))
bp_loss_fn = nn.CrossEntropyLoss()

# training loop
for epoch in range(HYPERPARAMS["epochs"]):
    dtp_model.train()
    bp_model.train()
    
    epoch_dtp_loss_top = 0.0
    epoch_dtp_f_loss = 0.0
    epoch_dtp_g_loss = 0.0
    epoch_dtp_correct = 0
    
    epoch_bp_loss = 0.0
    epoch_bp_correct = 0
    
    epoch_total_samples = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = torch.flatten(x, start_dim=1).to(device) # flatten from (batch_size, 1, 28, 28) to (batch_size, 784)
        y = y.to(device)
        
        # DTP training step
        dtp_loss_top_val, dtp_f_loss_val, dtp_g_loss_val, dtp_batch_correct = dtp_model.step(x=x, labels=y, optimizer=dtp_optimizer)
        
        epoch_dtp_loss_top += dtp_loss_top_val
        epoch_dtp_f_loss += dtp_f_loss_val
        epoch_dtp_g_loss += dtp_g_loss_val
        epoch_dtp_correct += dtp_batch_correct
        
        # BP training step
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
        
        if batch_idx % HYPERPARAMS["log_interval"] == 0:
            wandb.log({
                "dtp_loss_top_batch": dtp_loss_top_val,
                "dtp_f_loss_batch": dtp_f_loss_val,
                "dtp_g_loss_batch": dtp_g_loss_val,
                "dtp_train_batch_correct_count": dtp_batch_correct,
                "dtp_train_batch_acc": dtp_batch_correct / x.size(0),
                "bp_loss_batch": bp_loss_val.item(),
                "bp_train_batch_correct_count": bp_batch_correct,
                "bp_train_batch_acc": bp_batch_correct / x.size(0),
            })
            
    epoch_dtp_loss_top /= len(train_loader)
    epoch_dtp_f_loss /= len(train_loader)
    epoch_dtp_g_loss /= len(train_loader)
    epoch_dtp_acc = epoch_dtp_correct / epoch_total_samples

    epoch_bp_loss /= len(train_loader)
    epoch_bp_acc = epoch_bp_correct / epoch_total_samples

    print(f"Epoch {epoch}/{HYPERPARAMS['epochs']}:")
    print(f"  DTP: TopLoss={epoch_dtp_loss_top:.4f}, F-Loss={epoch_dtp_f_loss:.4f}, G-Loss={epoch_dtp_g_loss:.4f}, TrainAcc={epoch_dtp_acc*100:.2f}%")
    print(f"  BP:  Loss={epoch_bp_loss:.4f}, TrainAcc={epoch_bp_acc*100:.2f}%")
    
    wandb.log({
        "epoch": epoch,
        "dtp_epoch_loss_top": epoch_dtp_loss_top,
        "dtp_epoch_f_loss": epoch_dtp_f_loss,
        "dtp_epoch_g_loss": epoch_dtp_g_loss,
        "dtp_train_epoch_acc": epoch_dtp_acc,
        "bp_epoch_loss": epoch_bp_loss,
        "bp_train_epoch_acc": epoch_bp_acc,
    })
    
    # save checkpoint every N epochs or on the last epoch
    if (epoch + 1) % HYPERPARAMS["checkpoint_interval"] == 0 or epoch == HYPERPARAMS["epochs"] - 1:
        dtp_ckpt_path = os.path.join(checkpoint_path, f"dtp_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": dtp_model.state_dict(),
            "optimizer_state_dict": dtp_optimizer.state_dict(),
            "cfg": HYPERPARAMS
        }, dtp_ckpt_path)
        print(f"DTP Checkpoint saved: {dtp_ckpt_path}")

        bp_ckpt_path = os.path.join(checkpoint_path, f"bp_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": bp_model.state_dict(),
            "optimizer_state_dict": bp_optimizer.state_dict(),
            "cfg": HYPERPARAMS
        }, bp_ckpt_path)
        print(f"BP Checkpoint saved: {bp_ckpt_path}")

wandb.finish()
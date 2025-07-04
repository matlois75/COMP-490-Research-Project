import torch, argparse, glob, os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from Difference_Target_Propagation.dtp_network import DTPNetwork

def find_latest_checkpoint(checkpoint_dir="dtp_checkpoints", dataset=None):
    if dataset:
        pattern = os.path.join(checkpoint_dir, "**", f"run_dtp_{dataset}_*", "dtp_epoch_*.pt")
    else:
        pattern = os.path.join(checkpoint_dir, "**", "dtp_epoch_*.pt")
    ckpt_files = glob.glob(pattern, recursive=True)
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint files found")
    return max(ckpt_files, key=os.path.getmtime)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str)
    p.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'])
    args = p.parse_args()

    if args.ckpt:
        ckpt_dir = os.path.join("dtp_checkpoints", args.ckpt)
        pattern = os.path.join(ckpt_dir, "dtp_epoch_*.pt")
        ckpt_files = glob.glob(pattern)
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
        ckpt_path = max(ckpt_files, key=os.path.getmtime)
    else:
        ckpt_path = find_latest_checkpoint(dataset=args.dataset)
        print(f"Using latest checkpoint: {ckpt_path}")

    # load checkpoint and get the original training config
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    cfg = checkpoint['cfg']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.dataset == 'mnist':
        test_ds = MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == 'cifar10':
        test_ds = CIFAR10(root='data/', train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == 'cifar100':
        test_ds = CIFAR100(root='data/', train=False, download=True, transform=transforms.ToTensor())
        cfg["output_dim"] = 100
    
    test_ld = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    model = DTPNetwork(
        cfg[args.dataset]['input_dim'],
        tuple(cfg[args.dataset]['hidden_dims']),
        cfg['output_dim'],
        cfg['eta_hat'],
        cfg['sigma'],
        cfg.get('use_l_drl', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_ld:
            x = torch.flatten(x, start_dim=1).to(device)
            y = y.to(device)
            
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f'Test accuracy: {100 * correct/total:.2f}%')

if __name__ == '__main__':
    main()

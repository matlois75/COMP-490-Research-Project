import torch, argparse, glob, os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from Difference_Target_Propagation.dtp_network import DTPNetwork

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    pattern = os.path.join(checkpoint_dir, "**", "*.pt")
    ckpt_files = glob.glob(pattern, recursive=True)
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint files found")
    return max(ckpt_files, key=os.path.getmtime)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str)
    args = p.parse_args()

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = find_latest_checkpoint()
        print(f"Using latest checkpoint: {ckpt_path}")

    # load checkpoint and get the original training config
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    cfg = checkpoint['cfg']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_ds = MNIST(root=cfg.get('data_dir', 'data'), train=False, download=True, transform=ToTensor())
    test_ld = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 0))

    model = DTPNetwork(
        cfg['input_dim'],
        cfg['hidden_dims'],
        cfg['output_dim'],
        cfg['eta_hat'],
        cfg['sigma']
    )
    model.load_state_dict(checkpoint['model'])
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

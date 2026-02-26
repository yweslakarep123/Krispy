"""
Training FlowPolicy pada Kitchen dataset.
Menggunakan Consistency Flow Matching (CFM) loss dan arsitektur velocity field.
"""
import os
import argparse
import json
import numpy as np
import torch
from torch.optim import Adam

from tqdm import tqdm
from kitchen_dataset import get_dataloaders
from flow_policy import (
    FlowPolicyVelocityNet,
    FlowPolicyModel,
    compute_cfm_loss,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, state_dim, action_dim = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        use_seq=args.use_seq,
        use_mask=args.use_mask,
        synthetic_if_missing=args.synthetic,
        use_mjl_if_no_npy=not args.no_mjl,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )
    print(f"state_dim={state_dim}, action_dim={action_dim}")

    velocity_net = FlowPolicyVelocityNet(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
    ).to(device)

    flowpolicy_model = FlowPolicyModel(
        velocity_net=velocity_net,
        action_dim=action_dim,
        num_inference_steps=args.num_inference_steps,
        eps=args.cfm_eps,
    ).to(device)

    total_params = sum(p.numel() for p in velocity_net.parameters())
    trainable_params = sum(p.numel() for p in velocity_net.parameters() if p.requires_grad)
    print(f"FlowPolicy Velocity Network - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = Adam(velocity_net.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, args.epochs + 1):
        velocity_net.train()
        epoch_train_losses = []
        for states, actions in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
            leave=False,
        ):
            states = states.to(device)
            actions = actions.to(device)
            loss = compute_cfm_loss(
                velocity_net,
                states,
                actions,
                device,
                eps=args.cfm_eps,
                delta=args.cfm_delta,
                num_segments=args.cfm_num_segments,
                boundary=args.cfm_boundary,
                alpha=args.cfm_alpha,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        velocity_net.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for states, actions in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{args.epochs} [Val]",
                leave=False,
            ):
                states = states.to(device)
                actions = actions.to(device)
                loss = compute_cfm_loss(
                    velocity_net,
                    states,
                    actions,
                    device,
                    eps=args.cfm_eps,
                    delta=args.cfm_delta,
                    num_segments=args.cfm_num_segments,
                    boundary=args.cfm_boundary,
                    alpha=args.cfm_alpha,
                )
                epoch_val_losses.append(loss.item())

        avg_train = np.mean(epoch_train_losses)
        avg_val = np.mean(epoch_val_losses)
        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, "flow_policy_best.pt")
            torch.save(
                {
                    "velocity_net_state_dict": velocity_net.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                    "time_embed_dim": args.time_embed_dim,
                    "num_inference_steps": args.num_inference_steps,
                    "cfm_eps": args.cfm_eps,
                },
                ckpt_path,
            )
            print(f"  -> Model saved (val_loss improved to {avg_val:.6f})")

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"EPOCH {epoch}/{args.epochs} - Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

    # Simpan metrik
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"train_loss": train_loss_history, "val_loss": val_loss_history},
            f,
            indent=2,
        )
    print(f"Training metrics saved to {metrics_path}")

    # Plot training curves (opsional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Consistency Flow Matching Loss")
        plt.title("FlowPolicy Training Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(args.save_dir, "flowpolicy_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Training curves saved to {plot_path}")
    except Exception as e:
        print(f"Skip plot: {e}")

    print(f"Training selesai. Best val_loss={best_val_loss:.6f}")
    return best_val_loss


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="kitchen")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    # FlowPolicy architecture
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--time_embed_dim", type=int, default=128)
    p.add_argument("--num_inference_steps", type=int, default=10)
    # CFM loss
    p.add_argument("--cfm_eps", type=float, default=1e-2)
    p.add_argument("--cfm_delta", type=float, default=1e-2)
    p.add_argument("--cfm_num_segments", type=int, default=2)
    p.add_argument("--cfm_boundary", type=int, default=1)
    p.add_argument("--cfm_alpha", type=float, default=1e-5)
    # Data
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--use_seq", action="store_true")
    p.add_argument("--no_mask", action="store_true", help="Jangan pakai existence_mask")
    p.add_argument("--no_mjl", action="store_true", help="Jangan fallback ke file .mjl jika .npy tidak ada")
    p.add_argument("--synthetic", action="store_true", help="Pakai data sintetis jika .npy/.mjl tidak ada")
    p.add_argument("--state_dim", type=int, default=59, help="Dimensi state (untuk load .mjl / synthetic)")
    p.add_argument("--action_dim", type=int, default=9, help="Dimensi aksi (untuk load .mjl / synthetic)")
    p.add_argument("--log_every", type=int, default=5)
    args = p.parse_args()
    args.use_mask = not args.no_mask
    train(args)

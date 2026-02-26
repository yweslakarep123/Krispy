"""
Simulasi dan evaluasi FlowPolicy pada Kitchen dataset.
FlowPolicyModel: forward(state) -> action via ODE integration dari noise.
Metrik: MSE, MAE, R².
"""
import os
import argparse
import numpy as np
import torch
from torch.nn import MSELoss

from kitchen_dataset import (
    KitchenDataset,
    KitchenDatasetFromArrays,
    create_synthetic_kitchen,
)
from flow_policy import FlowPolicyVelocityNet, FlowPolicyModel


def load_policy(ckpt_path, device):
    """Load checkpoint dan bangun FlowPolicyModel."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    hidden_dim = ckpt.get("hidden_dim", 512)
    time_embed_dim = ckpt.get("time_embed_dim", 128)
    num_inference_steps = ckpt.get("num_inference_steps", 10)
    cfm_eps = ckpt.get("cfm_eps", 1e-2)

    velocity_net = FlowPolicyVelocityNet(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
    )
    velocity_net.load_state_dict(ckpt["velocity_net_state_dict"])
    model = FlowPolicyModel(
        velocity_net=velocity_net,
        action_dim=action_dim,
        num_inference_steps=num_inference_steps,
        eps=cfm_eps,
    )
    return model.to(device).eval(), state_dim, action_dim


def evaluate(model, obs, actions, device, criterion):
    """Hitung MSE, MAE, R² antara prediksi dan target."""
    with torch.no_grad():
        pred = model(obs.to(device))
    pred = pred.cpu()
    actions_np = actions.numpy()
    pred_np = pred.numpy()

    mse = criterion(pred, actions).item()
    mae = torch.abs(pred - actions).mean().item()
    ss_tot = ((actions_np - actions_np.mean(axis=0)) ** 2).sum()
    ss_res = ((actions_np - pred_np) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-12))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pred": pred_np,
        "target": actions_np,
    }


def run_simulation(model, dataset, device, max_steps=None):
    """Simulasi: prediksi aksi untuk setiap state di dataset."""
    model.eval()
    obs = dataset.obs
    actions = dataset.actions
    if max_steps is not None:
        obs = obs[:max_steps]
        actions = actions[:max_steps]
    return evaluate(model, obs, actions, device, MSELoss())


def print_report(metrics, title="Evaluation"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="kitchen")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flow_policy_best.pt")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_results", type=str, default="evaluation_results")
    parser.add_argument("--use_seq", action="store_true")
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint tidak ditemukan: {args.checkpoint}. Jalankan train.py dulu.")
        return

    model, _, _ = load_policy(args.checkpoint, device)
    if args.synthetic:
        obs, actions = create_synthetic_kitchen()
        dataset = KitchenDatasetFromArrays(obs, actions)
    else:
        dataset = KitchenDataset(args.data_dir, use_seq=args.use_seq, use_mask=not args.no_mask)

    metrics = run_simulation(model, dataset, device, max_steps=args.max_steps)
    print_report(metrics, title="Simulation & Evaluation - FlowPolicy on Kitchen")

    os.makedirs(args.save_results, exist_ok=True)
    out_path = os.path.join(args.save_results, "flow_policy_metrics.txt")
    with open(out_path, "w") as f:
        f.write("FlowPolicy on Kitchen Dataset - Simulation & Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE:  {metrics['mse']:.6f}\n")
        f.write(f"MAE:  {metrics['mae']:.6f}\n")
        f.write(f"R²:   {metrics['r2']:.6f}\n")
    print(f"Metrik disimpan ke {out_path}")

    np.savez(
        os.path.join(args.save_results, "flow_policy_predictions.npz"),
        predictions=metrics["pred"],
        targets=metrics["target"],
    )
    print("Prediksi dan target disimpan ke flow_policy_predictions.npz")


if __name__ == "__main__":
    main()

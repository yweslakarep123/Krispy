"""
Simulasi dan evaluasi FlowPolicy pada Kitchen dataset.
FlowPolicyModel: forward(state) -> action via ODE integration dari noise.
Metrik: MSE, MAE, R². Opsional: rekam video rollout (max timesteps) dan download.
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

# Default max timestep untuk video rollout
DEFAULT_VIDEO_MAX_TIMESTEPS = 280


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


def record_rollout_video(
    model,
    dataset,
    device,
    save_path: str,
    max_timesteps: int = 280,
    fps: int = 10,
    dpi: int = 100,
    state_plot_dims: int = 8,
    action_plot_dims: int = 9,
):
    """
    Rekam video hasil pengujian FlowPolicy: rollout max_timesteps step.
    Setiap frame menampilkan state (subset) dan aksi prediksi vs target hingga timestep t.
    Menyimpan ke save_path (mp4 atau gif).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    obs = dataset.obs[:max_timesteps]
    actions_gt = dataset.actions[:max_timesteps]
    T = len(obs)
    if T == 0:
        print("Tidak ada data untuk video.")
        return None

    with torch.no_grad():
        preds = model(obs.to(device)).cpu().numpy()
    obs_np = obs.numpy()
    actions_gt_np = actions_gt.numpy()

    state_plot_dims = min(state_plot_dims, obs_np.shape[1])
    action_plot_dims = min(action_plot_dims, preds.shape[1])

    ext = os.path.splitext(save_path)[1].lower()
    use_mp4 = ext == ".mp4"
    frames = []

    for t in range(T):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
        fig.suptitle(f"FlowPolicy Rollout — timestep {t + 1}/{T}", fontsize=12)

        ax1.set_title("State (subset)")
        ax1.bar(range(state_plot_dims), obs_np[t, :state_plot_dims], color="steelblue", alpha=0.8)
        ax1.set_ylabel("state")
        ax1.set_xlim(-0.5, state_plot_dims - 0.5)

        ax2.set_title("Action: pred vs target")
        x = np.arange(action_plot_dims)
        w = 0.35
        ax2.bar(x - w / 2, preds[t, :action_plot_dims], w, label="pred", color="green", alpha=0.8)
        ax2.bar(x + w / 2, actions_gt_np[t, :action_plot_dims], w, label="target", color="orange", alpha=0.8)
        ax2.set_ylabel("action")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_xlim(-0.5, action_plot_dims - 0.5)

        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    if use_mp4:
        try:
            import imageio.v3 as iio
            iio.imwrite(save_path, frames, fps=fps, codec="libx264", quality=8)
        except Exception as e:
            try:
                import imageio
                imageio.mimsave(save_path, frames, fps=fps)
            except Exception as e2:
                save_path = save_path.replace(".mp4", ".gif")
                import imageio
                imageio.mimsave(save_path, frames, fps=fps, loop=0)
                print(f"MP4 gagal, simpan sebagai GIF: {save_path} ({e})")
        else:
            print(f"Video disimpan: {save_path}")
    else:
        try:
            import imageio.v3 as iio
            iio.imwrite(save_path, frames, fps=fps, loop=0)
        except Exception:
            import imageio
            imageio.mimsave(save_path, frames, fps=fps, loop=0)
        print(f"Video disimpan: {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="kitchen")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flow_policy_best.pt")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_results", type=str, default="evaluation_results")
    parser.add_argument("--use_seq", action="store_true")
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--record_video", action="store_true", help="Rekam video rollout hasil pengujian")
    parser.add_argument("--max_timesteps", type=int, default=DEFAULT_VIDEO_MAX_TIMESTEPS, help="Max timestep untuk video (default 280)")
    parser.add_argument("--video_path", type=str, default=None, help="Path simpan video (default: save_results/flowpolicy_rollout_280.mp4)")
    parser.add_argument("--video_fps", type=int, default=10)
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

    if args.record_video:
        video_path = args.video_path or os.path.join(
            args.save_results, f"flowpolicy_rollout_{args.max_timesteps}.mp4"
        )
        record_rollout_video(
            model,
            dataset,
            device,
            save_path=video_path,
            max_timesteps=args.max_timesteps,
            fps=args.video_fps,
        )
        print(f"Video hasil pengujian (max {args.max_timesteps} timesteps) dapat didownload dari: {os.path.abspath(video_path)}")


if __name__ == "__main__":
    main()

"""
Loader untuk Kitchen dataset: .npy dan .mjl.
- .npy: all_observations, all_actions, dll. di data_dir.
- .mjl: file log MuJoCo (playdata) di data_dir atau subfolder (mis. kitchen_demos_multitask/.../*.mjl).
"""
import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def find_mjl_files(data_dir: str):
    """Cari semua file .mjl di data_dir dan subfolder (rekursif)."""
    mjl_paths = []
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        return mjl_paths
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".mjl"):
                mjl_paths.append(os.path.join(root, f))
    return sorted(mjl_paths)


def _load_mjl_via_mujoco(path: str, state_dim: int = 59, action_dim: int = 9):
    """Load .mjl menggunakan API MuJoCo (Recorder / open_log) jika ada."""
    try:
        import mujoco
    except ImportError:
        return None, None
    # MuJoCo 3.x: beberapa versi punya recorder untuk baca log
    if hasattr(mujoco, "Recorder") and hasattr(mujoco.Recorder, "open"):
        try:
            with mujoco.Recorder.open(path, "r") as log:
                # Baca frame: format tergantung versi (biasanya time, state, ctrl)
                times, qpos_list, qvel_list, ctrl_list = [], [], [], []
                while True:
                    try:
                        frame = log.read_frame()
                        if frame is None:
                            break
                        times.append(getattr(frame, "time", 0.0))
                        qpos_list.append(np.array(getattr(frame, "qpos", [])))
                        qvel_list.append(np.array(getattr(frame, "qvel", [])))
                        ctrl_list.append(np.array(getattr(frame, "ctrl", [])))
                    except (StopIteration, EOFError):
                        break
                if not qpos_list:
                    return None, None
                qpos = np.array(qpos_list, dtype=np.float32)
                qvel = np.array(qvel_list, dtype=np.float32)
                ctrl = np.array(ctrl_list, dtype=np.float32)
                obs = np.concatenate([qpos, qvel], axis=-1)
                if obs.shape[1] != state_dim:
                    obs = obs[:, :state_dim] if obs.shape[1] >= state_dim else np.pad(obs, ((0, 0), (0, state_dim - obs.shape[1])))
                if ctrl.shape[1] != action_dim:
                    ctrl = ctrl[:, :action_dim] if ctrl.shape[1] >= action_dim else np.pad(ctrl, ((0, 0), (0, action_dim - ctrl.shape[1])))
                return obs, ctrl
        except Exception:
            pass
    # Alternatif: mujoco.load_log / open_log (nama bisa beda per versi)
    for name in ("load_log", "open_log", "recorder"):
        mod = getattr(mujoco, name, None)
        if mod is None and name == "recorder":
            try:
                mod = __import__("mujoco.recorder", fromlist=["Recorder"])
            except ImportError:
                pass
        if mod is not None and callable(getattr(mod, "open", None)):
            try:
                with getattr(mod, "open")(path, "r") as log:
                    obs_list, ctrl_list = [], []
                    for frame in getattr(log, "read", lambda: iter([]))() or log:
                        o = getattr(frame, "qpos", None)
                        if o is not None:
                            v = getattr(frame, "qvel", np.zeros_like(o))
                            obs_list.append(np.concatenate([np.ravel(o), np.ravel(v)])[:state_dim])
                        c = getattr(frame, "ctrl", None)
                        if c is not None:
                            ctrl_list.append(np.ravel(c)[:action_dim])
                    if obs_list and ctrl_list:
                        return np.array(obs_list, dtype=np.float32), np.array(ctrl_list, dtype=np.float32)
            except Exception:
                pass
    return None, None


def _load_mjl_raw(path: str, state_dim: int = 59, action_dim: int = 9):
    """Fallback: baca .mjl sebagai binary (format umum: header lalu time, qpos, qvel, ctrl per frame)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return None, None
    if len(raw) < 20:
        return None, None
    # Coba format MuJoCo log: 4 int (version, nq, nv, nu) lalu per frame: double time, lalu qpos, qvel, ctrl
    try:
        version, nq, nv, nu = struct.unpack("iiii", raw[:16])
        if version < 0 or version > 10 or nq > 1000 or nv > 1000 or nu > 500:
            nq, nv, nu = state_dim - 9, 9, action_dim  # guess kitchen-like
            offset = 0
        else:
            offset = 16
        nstate = nq + nv
        frame_size = 8 + 8 * nstate + 8 * nu  # time + qpos+qvel + ctrl
        n_frames = (len(raw) - offset) // frame_size
        if n_frames <= 0:
            return None, None
        obs_list = []
        ctrl_list = []
        for i in range(n_frames):
            start = offset + i * frame_size
            time = struct.unpack("d", raw[start : start + 8])[0]
            state = struct.unpack(f"{nstate}d", raw[start + 8 : start + 8 + 8 * nstate])
            ctrl = struct.unpack(f"{nu}d", raw[start + 8 + 8 * nstate : start + frame_size])
            obs_list.append(np.array(state[:state_dim] if nstate >= state_dim else list(state) + [0.0] * (state_dim - nstate), dtype=np.float32))
            ctrl_list.append(np.array(ctrl[:action_dim] if nu >= action_dim else list(ctrl) + [0.0] * (action_dim - nu), dtype=np.float32))
        return np.array(obs_list), np.array(ctrl_list)
    except Exception:
        pass
    return None, None


def load_kitchen_from_mjl(
    data_dir: str,
    state_dim: int = 59,
    action_dim: int = 9,
    max_trajs: int = 500,
):
    """
    Load semua trajectory dari file .mjl di data_dir (dan subfolder).
    Mengembalikan (obs, actions) dengan shape (total_steps, state_dim) dan (total_steps, action_dim).
    """
    mjl_paths = find_mjl_files(data_dir)
    if not mjl_paths:
        return None, None
    all_obs, all_actions = [], []
    for path in mjl_paths[:max_trajs]:
        obs, actions = _load_mjl_via_mujoco(path, state_dim, action_dim)
        if obs is None:
            obs, actions = _load_mjl_raw(path, state_dim, action_dim)
        if obs is not None and actions is not None and len(obs) > 0:
            all_obs.append(obs)
            all_actions.append(actions)
    if not all_obs:
        return None, None
    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    return obs, actions


def load_kitchen_data(data_dir: str):
    """Load semua array dari data_dir. Prioritas: .npy, lalu (jika tidak ada) tidak otomatis load .mjl."""
    data = {}
    for name in [
        "all_observations",
        "all_actions",
        "observations_seq",
        "actions_seq",
        "all_init_qpos",
        "all_init_qvel",
        "existence_mask",
    ]:
        path = os.path.join(data_dir, f"{name}.npy")
        if os.path.isfile(path):
            try:
                arr = np.load(path, allow_pickle=True)
                if hasattr(arr, 'shape'):
                    data[name] = arr
                else:
                    print(f"Skip {name}: not a numeric array (e.g. LFS pointer)")
            except Exception as e:
                print(f"Skip {name}: {e}")
    return data


def flatten_trajectories(obs, actions, mask=None):
    """
    Flatten (N, T, D) -> (N*T, D) atau (N, T, D) -> (total_valid, D) jika mask ada.
    """
    if obs.ndim == 2:
        return obs, actions
    n, t, do = obs.shape
    _, _, da = actions.shape
    if mask is not None:
        obs_flat = obs[mask].reshape(-1, do)
        act_flat = actions[mask].reshape(-1, da)
        return obs_flat, act_flat
    obs_flat = obs.reshape(-1, do)
    act_flat = actions.reshape(-1, da)
    return obs_flat, act_flat


class KitchenDatasetFromArrays(Dataset):
    """Dataset dari array obs (N,T,Do) dan actions (N,T,Da) sudah di-flatten."""

    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        if obs.ndim == 3:
            obs = obs.reshape(-1, obs.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class KitchenDataset(Dataset):
    """PyTorch Dataset untuk (observation, action) dari kitchen. Sumber: .npy atau .mjl."""

    def __init__(
        self,
        data_dir: str,
        use_seq=False,
        use_mask=True,
        use_mjl_if_no_npy=True,
        state_dim=59,
        action_dim=9,
    ):
        raw = load_kitchen_data(data_dir)
        obs, actions = None, None

        if raw:
            if use_seq and "observations_seq" in raw and "actions_seq" in raw:
                obs = raw["observations_seq"]
                actions = raw["actions_seq"]
            else:
                obs = raw.get("all_observations")
                actions = raw.get("all_actions")
            if obs is not None and actions is not None:
                mask = raw.get("existence_mask") if use_mask else None
                obs, actions = flatten_trajectories(obs, actions, mask)

        if obs is None or actions is None:
            if use_mjl_if_no_npy:
                mjl_paths = find_mjl_files(data_dir)
                if mjl_paths:
                    print(f"Load dari {len(mjl_paths)} file .mjl (tidak ada .npy valid).")
                    obs, actions = load_kitchen_from_mjl(data_dir, state_dim=state_dim, action_dim=action_dim)
            if obs is None or actions is None:
                raise FileNotFoundError(
                    f"Tidak ada data di {data_dir}. Sediakan .npy (all_observations, all_actions) atau .mjl di (sub)folder."
                )

        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def create_synthetic_kitchen(num_trajs=50, steps_per_traj=100, obs_dim=60, action_dim=9, seed=42):
    """Buat data sintetis mirip kitchen (N, T, D) untuk uji pipeline jika .npy belum di-pull."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((num_trajs, steps_per_traj, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((num_trajs, steps_per_traj, action_dim)).astype(np.float32)
    return obs, actions


def get_dataloaders(
    data_dir,
    batch_size=256,
    val_ratio=0.1,
    use_seq=False,
    use_mask=True,
    synthetic_if_missing=False,
    use_mjl_if_no_npy=True,
    state_dim=59,
    action_dim=9,
):
    """
    Train/val DataLoaders.
    - Prioritas: .npy (all_observations, all_actions) di data_dir.
    - Jika tidak ada .npy valid: load dari file .mjl di data_dir/subfolder (use_mjl_if_no_npy=True).
    - Jika tetap gagal dan synthetic_if_missing=True: pakai data sintetis.
    """
    try:
        ds = KitchenDataset(
            data_dir,
            use_seq=use_seq,
            use_mask=use_mask,
            use_mjl_if_no_npy=use_mjl_if_no_npy,
            state_dim=state_dim,
            action_dim=action_dim,
        )
    except (FileNotFoundError, KeyError) as e:
        if not synthetic_if_missing:
            raise
        print(f"Data tidak ditemukan ({e}). Menggunakan data sintetis untuk demonstrasi.")
        obs, actions = create_synthetic_kitchen(obs_dim=state_dim, action_dim=action_dim)
        ds = KitchenDatasetFromArrays(obs, actions)
    n = len(ds)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, ds.obs.shape[1], ds.actions.shape[1]

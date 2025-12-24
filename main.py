from run_traj import run_all_trajectories
import numpy as np

if __name__ == "__main__":
    alphas = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 9e-03, 3e-2, 5e-2, 6e-2, 1e-1]
    for a in alphas:
        OUT = run_all_trajectories(
            n_trajectories=1024,
            N=2000000,
            p=5,
            sigma=0.02,
            uniform_low=-np.sqrt(3),
            uniform_high=np.sqrt(3),
            base_seed=12345,
            alpha = a,
            num_workers=8,
            batch_size=4,
            out_path=f"sgd_1024_trajs_linreg_2000000_alpha_{a}_thetas.npz"
        )
        print("Saved to", OUT)
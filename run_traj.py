import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import os

def run_single_trajectory(traj_id, N, p, sigma,  theta_true,
                          uniform_low, uniform_high,
                          base_seed, alpha):

    seed = int(base_seed) + int(traj_id)
    rng = np.random.default_rng(seed)


    X = rng.uniform(low=uniform_low, high=uniform_high, size=(N, p))
    noise = rng.normal(0.0, sigma, size=N)
    y = X.dot(theta_true) + noise  

    
    theta_hat = np.zeros(p, dtype=float)    
 
    theta_hat_history = np.zeros((N + 1, p), dtype=float)


    theta_hat_history[0] = theta_hat.copy()
 

 
    for n in range(1, N + 1):
        xn = X[n - 1]          
        yn = y[n - 1]
        alpha_n = alpha

        residual = yn - xn.dot(theta_hat)
        grad = -2.0 * residual * xn  
        # print("alpha=", alpha, "grad=", grad)
        
        theta_hat = theta_hat - alpha_n * grad
        # print(theta_hat)
        theta_hat_history[n] = theta_hat.copy()


    return traj_id, theta_hat_history[999::10000]

def run_batch_trajectories(batch_ids, N, p, sigma, theta_true,
                           uniform_low, uniform_high,
                           base_seed, alpha):

    results = []
    for traj_id in batch_ids:
        results.append(run_single_trajectory(traj_id, N, p, sigma,
                                             theta_true,
                                             uniform_low, uniform_high,
                                             base_seed, alpha))
    return results

def run_all_trajectories(n_trajectories=1024,
                         N=1024000,
                         p=10,
                         sigma=0.01,
                         uniform_low=-np.sqrt(3),
                         uniform_high=np.sqrt(3),
                         base_seed=12345,
                         alpha=0.5,
                         num_workers=None,
                         batch_size = 16,
                         out_path="sgd_trajectories.npz"):
 
    mu = 1
    q = p // 2
    theta_true = np.concatenate([mu * np.ones(q), -(mu/10) * np.ones(p - q)])

    theta_hat_histories = np.zeros((n_trajectories, (N + 1)//10000, p), dtype=float)

    batch_ids_list = [list(range(i, min(i + batch_size, n_trajectories)))
                      for i in range(0, n_trajectories, batch_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {exe.submit(run_batch_trajectories, batch_ids, N, p, sigma,
                              theta_true, uniform_low, uniform_high,
                              base_seed, alpha): batch_ids
                   for batch_ids in batch_ids_list}


        for fut in as_completed(futures):
            batch_results = fut.result()
            print("one_finish")
            for traj_id, th_hat_hist in batch_results:
                theta_hat_histories[traj_id] = th_hat_hist



 
    np.savez_compressed(out_path,
                        theta_hat_histories=theta_hat_histories,
                        theta_true=theta_true,
                        N=N, p=p, sigma=sigma, n_trajectories=n_trajectories,
                        uniform_low=uniform_low, uniform_high=uniform_high,
                        base_seed=base_seed, alpha=alpha)
    return out_path

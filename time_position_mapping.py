#!/usr/bin/python3
"""
Single-task gpCAM autonomous experiment: 2D (position [mm], time [s]) space.

Adapted from:
  - multitask_server_Mar2025.py  (reachability constraint, cost function)
  - new_server.py                (single-task kernel, time-based cost)
  - multi_decision_mapping.py    (queue loop, file-based data wrangling)

Uses GPOptimizer (gpcam)  no tsuchinoko dependency.
"""

import numpy as np
import time
import sys
import os
import random
from colorama import Fore, Style
from pathlib import Path
from datetime import datetime
from scipy.optimize import NonlinearConstraint

from gpcam import GPOptimizer
from gpcam.kernels import get_distance_matrix, matern_kernel_diff1

bluesky_PATH = '/nsls2/data/cms/shared/config/bluesky/profile_collection/users/2026-1/KChen-Wiegart/2026C1/'
queue_PATH   = '/nsls2/data/cms/shared/config/bluesky/profile_collection/users/2026-1/KChen-Wiegart/2026C1/'
queue_PATH in sys.path or sys.path.append(queue_PATH)
from CustomQueue import Queue_decision

filename_format = '%Y-%m-%d_%H-%M-%S'

Abs_Path      = '/nsls2/data/cms/proposals/2026-1/pass-319051/experiments/2_PTA/data/'
CMSsaves_dir  = os.path.join(Abs_Path, 'CMSsaves/')
CMStotals_dir = os.path.join(Abs_Path, 'CMStotals/')


##################################################
# Experimental parameters
##################################################

v                = 3.0        # motor speed [mm/s]
end_of_time      = 2*60*60.   # experiment duration [s]
time_buffer      = 5.         # buffer for GP optimization time [s]
measurement_cost = 15         # time per measurement including alignment [s]

# Bounds: x in [0, 30] mm, time in [0, 2*end_of_time] s
# Time upper bound extends to 2x for GP stability (experiment is killed at end_of_time)
bounds = np.array([[1., 30.],
                   [0., 2. * end_of_time]])

init_N = 5 # number of random measurements before GP kicks in


##################################################
# GP hyperparameters
##################################################

# [signal_variance, length_scale_x (mm), length_scale_time (s)]
hps_initial = np.array([1e3, 2., 5 * 60.])

hps_bounds = np.array([
    [1e-3,  1e7],              # signal variance
    [0.1,   40.],              # length scale in x [mm]
    [0.1,   10. * end_of_time] # length scale in time [s]
])

# Iteration indices at which to retrain hyperparameters
training_at = [10, 20, 30]


##################################################
# Kernel
##################################################

def kernel(x1, x2, hps):
    """Anisotropic Mat�rn-5/2 kernel with separate length scales for x and time."""
    x1_scaled = np.column_stack([x1[:, 0] / hps[1], x1[:, 1] / hps[2]])
    x2_scaled = np.column_stack([x2[:, 0] / hps[1], x2[:, 1] / hps[2]])
    d = get_distance_matrix(x1_scaled, x2_scaled)
    return hps[0] * matern_kernel_diff1(d, 1.)


##################################################
# Cost function
##################################################

def cost(origin, x, arguments=None):
    """
    Time-based cost: waiting time + measurement overhead.
    Points in the past (time < current) are assigned a prohibitive cost.
    """
    t = np.abs(x[:, 1] - origin[1]) + measurement_cost
    past = np.where(origin[1] > x[:, 1])
    t[past] = end_of_time * 10.
    return t


##################################################
# Reachability constraint
##################################################

def make_constraint(my_gp):
    """
    Returns a NonlinearConstraint enforcing that the motor can physically
    reach position x from the last measured position within the available time.

    g(x) >= 0  iff the point (pos, t) is reachable:
        (dt * v)^2 - (dx)^2 >= 0   AND   dt > time_buffer
    where dt = t - (t_current + time_buffer), dx = pos - x_current.
    """
    current_time = float(my_gp.x_data[-1, 1])
    current_x    = float(my_gp.x_data[-1, 0])

    def g(x):
        # x   = np.atleast_1d(np.asarray(x, dtype=float))
        # pos = x[0]
        # t   = x[1]
        # dt  = t - (current_time + time_buffer)
        # val = (dt * v) ** 2 - (pos - current_x) ** 2
        # if dt < 0. or val < 0.:
        #     return -1.
        # return float(np.sqrt(val))

        # Cheng-Chu: scipy pass scalar path or vectorized path
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            # Single point [pos, t]  return shape (1,)
            pos = x[0]
            t   = x[1]
            dt  = t - (current_time + time_buffer)
            val = (dt * v) ** 2 - (pos - current_x) ** 2
            result = -1. if (dt < 0. or val < 0.) else float(np.sqrt(val))
            return np.array([result])
        else:
            # Batch from scipy differential_evolution: shape (2, n_pop)
            # Must return shape (1, n_pop) = (M, S)
            pos = x[0]
            t   = x[1]
            dt  = t - (current_time + time_buffer)
            val = (dt * v) ** 2 - (pos - current_x) ** 2
            result = np.where((dt < 0.) | (val < 0.), -1., np.sqrt(np.maximum(val, 0.)))
            return result[np.newaxis, :]  # shape (1, n_pop)

    return NonlinearConstraint(g, 0., np.inf)


##################################################
# Data wrangling
##################################################

def wait_for_stable_file(file_path, wait_time=2):
    """Wait until the file has not been modified for at least `wait_time` seconds."""
    file = Path(file_path)
    if not file.exists():
        print(f"File {file_path} does not exist.")
        return
    while True:
        elapsed = time.time() - file.stat().st_mtime
        if elapsed > wait_time:
            break
        print(f"Waiting... File was modified {elapsed:.2f} s ago.")
        time.sleep(0.5)


def get_all_data():
    """Load all CMS data from the totals directory. Returns empty arrays if no data yet."""
    x_data, y_data, variance_data = [], [], []

    inpath = Path(CMStotals_dir)
    files  = list(inpath.glob("*.npy"))
    if not files:
        print("get_all_data: no data files found.")
        return np.array([]), np.array([]), np.array([])

    last_file = max(files, key=lambda f: datetime.strptime(f.stem, filename_format))
    wait_for_stable_file(last_file, wait_time=2)
    CMS_data = np.load(last_file, allow_pickle=True)
    print(f"get_all_data: loaded {len(CMS_data)} points from {last_file.name}")

    for point in CMS_data:
        x_data.append([point['position'][0], point['position'][1]])  # (x_mm, time_s)
        y_data.append(point['value'][0])
        variance_data.append(point['variance'][0])

    return np.asarray(x_data), np.asarray(y_data), np.asarray(variance_data)


def saveCMS(data):
    dt_str = datetime.now().strftime(filename_format)

    # Save individual measurement result
    outfile = f"{CMSsaves_dir}{dt_str}.npy"
    print(f"saveCMS: saving to {outfile}")
    np.save(outfile, data, allow_pickle=True)

    # Append to cumulative totals file
    inpath = Path(CMStotals_dir)
    files  = list(inpath.glob("*.npy"))

    if not files:
        total_data = [data[0]]
    else:
        last_file  = max(files, key=lambda f: datetime.strptime(f.stem, filename_format))
        total_data = np.load(last_file, allow_pickle=True)
        total_data = np.append(total_data, data[0])

    outfile = f"{CMStotals_dir}{dt_str}.npy"
    np.save(outfile, total_data, allow_pickle=True)


##################################################
# Build GP
##################################################

def build_GP(x_data, y_data, variance_data=None):
    if variance_data is not None:
        variance_data = np.asarray(variance_data, dtype=np.float64).flatten()
        print(f"build_GP: variance_data shape={variance_data.shape}")

    print(f"build_GP: {len(x_data)} points, x={x_data}, y={y_data}")

    my_gp = GPOptimizer(
        x_data, y_data,
        init_hyperparameters = hps_initial,
        noise_variances      = variance_data,
        kernel_function   = kernel,
        cost_function        = cost,
    )

    print("build_GP: training...")
    my_gp.train(hyperparameter_bounds=hps_bounds, max_iter=100)
    print(f"build_GP: hyperparameters = {my_gp.get_hyperparameters()}")

    return my_gp


##################################################
# AE logic
##################################################

def initial_AE():
    """Pick a random starting position. Time = 0 (beamline records actual elapsed time)."""
    x_pos = np.random.uniform(bounds[0, 0], bounds[0, 1])
    new_result = {'position': [x_pos, 0.], 'measured': False, 'analyzed': False}
    return [new_result]


def nominal_AE(data, my_gp, N):
    """Tell the latest measurement to the GP, then ask for the next position."""

    new_x        = [[data[0]['position'][0], data[0]['position'][1]]]
    new_y        = [data[0]['value'][0]]
    new_variance = [data[0]['variance'][0]]

    print(f"nominal_AE (N={N}): telling point {new_x}, value {new_y}")
    my_gp.tell(np.asarray(new_x), np.asarray(new_y),
               noise_variances=np.asarray(new_variance), append=True)

    if N in training_at:
        print('=' * 40)
        print(f"nominal_AE (N={N}): retraining...")
        print(f"{Fore.CYAN}GP data:\n{my_gp.x_data}{Style.RESET_ALL}")
        print(f"Hyperparameters: {my_gp.get_hyperparameters()}")
        my_gp.train(hyperparameter_bounds=hps_bounds, max_iter=100)
        print('=' * 40)

    # Rebuild constraint from the latest GP data point
    nlc = make_constraint(my_gp)

    suggestion = my_gp.ask(
        bounds,
        n                    = 1,
        acquisition_function = 'variance',
        method               = 'global',
        constraints          = (nlc,),
    )

    print(f"nominal_AE (N={N}): suggestion = {suggestion}")
    position   = suggestion['x'][0]
    new_result = {'position': list(position), 'measured': False, 'analyzed': False}
    return [new_result]


##################################################
# Main loop
##################################################

def loop(queue, N_max=20000):

    # Send initial random measurement to beamline
    data = initial_AE()
    queue.publish(data)

    my_gp = None
    N     = 1

    while N < N_max:

        print(f"\n{Fore.RED}==== Loop (N={N}): waiting for data ===={Style.RESET_ALL}")
        data = queue.get()
        print(f"{Fore.RED}==== Loop (N={N}): received ===={Style.RESET_ALL}")
        print(data)

        doCMSsave = True
        if N < init_N:
            # Still in random-initialization phase
            print("cms data save")
            saveCMS(data) #Carly
            data = initial_AE()

        else:
            if my_gp is None:
                saveCMS(data)
                doCMSsave = False
                # First GP build: load all previously saved data (current point NOT yet saved)
                # so get_all_data() does not see the current point  avoids duplicate tell()
                x_data, y_data, variance_data = get_all_data()
                my_gp = build_GP(x_data, y_data, variance_data)
                print(f'After 1st build :{x_data}')

            # Save current point AFTER building GP, then tell it via nominal_AE
            if doCMSsave:
                saveCMS(data)
            data = nominal_AE(data, my_gp, N)
            print(f'After 1st suggestion : {data}')
        queue.publish(data)
        print(f"{Fore.RED}==== Loop (N={N}): published next position ===={Style.RESET_ALL}")
        N += 1


##################################################
# Entry point
##################################################

if __name__ == "__main__":

    print('=' * 50)
    print('= DO YOU NEED TO ./flush.sh ?')
    print('=' * 50)
    time.sleep(4)

    q = Queue_decision()
    loop(q)

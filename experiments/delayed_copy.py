import argparse
import math
import os
import warnings

import numpy as np
import optuna
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import DeepReservoir, UnicycleReservoir


def parse_delays(spec):
    delays = []
    for item in spec.split(","):
        if not item.strip():
            continue
        parts = [int(x) for x in item.split(":")]
        if len(parts) == 1:
            delays.append(parts[0])
        elif len(parts) in (2, 3):
            start, stop = parts[:2]
            step = parts[2] if len(parts) == 3 else 1
            delays.extend(range(start, stop + 1, step))
        else:
            raise ValueError(f"Invalid delay spec: {item}")
    delays = sorted(set(delays))
    if not delays or min(delays) < 0:
        raise ValueError("Delays must be non-negative.")
    return delays


def make_distractor(delay, args, rng):
    if delay == 0:
        return np.empty(0, dtype=np.float32)
    distractor = rng.normal(0.0, args.distractor_scaling, delay).astype(np.float32)

    return distractor


def make_batch(delay, args, rng):
    x = np.zeros((args.batch_size, delay + 2, 1), dtype=np.float32)
    y = np.zeros(args.batch_size, dtype=np.int64)

    for start in range(0, args.batch_size, args.n_symbols):
        stop = min(start + args.n_symbols, args.batch_size)
        symbols = np.arange(stop - start, dtype=np.int64)
        y[start:stop] = symbols
        x[start:stop, 0, 0] = -args.symbol_scaling * (symbols + 1)
        x[start:stop, 1 : delay + 1, 0] = make_distractor(delay, args, rng)
        x[start:stop, delay + 1, 0] = args.cue_scaling
    x = torch.from_numpy(x)
    return x, y


def make_input_map(n_units, num_non_zero, min_value, max_value, device):
    input_map = torch.zeros(1, n_units, device=device)
    num_non_zero = min(max(num_non_zero, 0), n_units)
    if num_non_zero == 0:
        return input_map

    non_zero_indices = torch.randperm(n_units, device=device)[:num_non_zero]
    values = torch.rand(num_non_zero, device=device) * (max_value - min_value)
    input_map[0, non_zero_indices] = values + min_value
    return input_map


def move_static_tensors(model, device):
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    for name in ("lin_damping", "ang_damping", "mass_vector", "j_vector"):
        value = getattr(model.unicycle_network, name).to(device)
        setattr(model.unicycle_network, name, value)


def suggest_unicycle_config(trial, n_units):
    lin_stiff_min = trial.suggest_float("lin_stiff_min", 0.1, 1.0)
    lin_stiff_max = trial.suggest_float("lin_stiff_max", lin_stiff_min, 10.0)
    ang_stiff_min = trial.suggest_float("ang_stiff_min", 0.1, 1.0)
    ang_stiff_max = trial.suggest_float("ang_stiff_max", ang_stiff_min, 2.0)
    lin_damping_min = trial.suggest_float("lin_damping_min", 0.1, 1.0)
    lin_damping_max = trial.suggest_float("lin_damping_max", lin_damping_min, 5.0)
    ang_damping_min = trial.suggest_float("ang_damping_min", 0.1, 10.0)
    ang_damping_max = trial.suggest_float("ang_damping_max", ang_damping_min, 20.0)
    dt = trial.suggest_float("dt", 0.0001, 0.01, step=0.0005)
    inp_bias = trial.suggest_float("inp_bias", -1.0, 1.0)
    anchor_con_fraction = trial.suggest_float(
        "anchor_con_fraction", 0.0, 1.0, step=0.1
    )

    lin_input_non_zero = trial.suggest_int("non_zero_elements", 1, n_units)
    lin_input_min = trial.suggest_float("magnitude_min", -10.0, 0.0)
    lin_input_max = trial.suggest_float("magnitude_max", lin_input_min, 20.0)

    ang_input_non_zero = trial.suggest_int("non_zero_elements_ang", 1, n_units)
    ang_input_min = trial.suggest_float("magnitude_min_ang", -10.0, 0.0)
    ang_input_max = trial.suggest_float("magnitude_max_ang", ang_input_min, 10.0)

    n_connections_fraction = trial.suggest_float(
        "n_connections_fraction", 0.2, 1.0, step=0.1
    )
    washup_steps = trial.suggest_int("washup_steps", 0, 4000, step=1000)
    n_connections_ang_fraction = trial.suggest_float(
        "n_connections_ang_fraction", 0.2, 1.0, step=0.1
    )
    anchor_con_fraction_ang = trial.suggest_float(
        "anchor_con_fraction_ang", 0.0, 1.0, step=0.1
    )

    eq_dist_min = trial.suggest_float("eq_dist_min", 0.2, 1.0)
    eq_dist_max = trial.suggest_float("eq_dist_max", eq_dist_min, 2.0)
    eq_dist_min_ang = trial.suggest_float("eq_dist_min_ang", -2 * math.pi, 0.0)
    eq_dist_max_ang = trial.suggest_float(
        "eq_dist_max_ang", eq_dist_min_ang, 2 * math.pi
    )

    return {
        "model": "unicycle",
        "n_units": n_units,
        "lin_stiff_min": lin_stiff_min,
        "lin_stiff_max": lin_stiff_max,
        "ang_stiff_min": ang_stiff_min,
        "ang_stiff_max": ang_stiff_max,
        "lin_damping_min": lin_damping_min,
        "lin_damping_max": lin_damping_max,
        "ang_damping_min": ang_damping_min,
        "ang_damping_max": ang_damping_max,
        "dt": dt,
        "inp_bias": inp_bias,
        "anchor_con_fraction": anchor_con_fraction,
        "lin_input_non_zero": lin_input_non_zero,
        "lin_input_min": lin_input_min,
        "lin_input_max": lin_input_max,
        "ang_input_non_zero": ang_input_non_zero,
        "ang_input_min": ang_input_min,
        "ang_input_max": ang_input_max,
        "n_connections_fraction": n_connections_fraction,
        "washup_steps": washup_steps,
        "n_connections_ang_fraction": n_connections_ang_fraction,
        "anchor_con_fraction_ang": anchor_con_fraction_ang,
        "n_past_steps_readout": 0,
        "eq_dist_min": eq_dist_min,
        "eq_dist_max": eq_dist_max,
        "eq_dist_min_ang": eq_dist_min_ang,
        "eq_dist_max_ang": eq_dist_max_ang,
    }


def suggest_deep_reservoir_config(trial, n_units):
    return {
        "model": "deep_reservoir",
        "n_units": n_units,
        "input_scaling": trial.suggest_float("input_scaling", 0.01, 10.0, log=True),
        "leaky": trial.suggest_float("leaky", 0.1, 1.0),
    }


def suggest_config(trial, args):
    if args.model == "deep_reservoir":
        return suggest_deep_reservoir_config(trial, args.n_hid)
    return suggest_unicycle_config(trial, args.n_hid)


def unicycle_config_from_params(params, n_units):
    return {
        "model": "unicycle",
        "n_units": n_units,
        "lin_stiff_min": params["lin_stiff_min"],
        "lin_stiff_max": params["lin_stiff_max"],
        "ang_stiff_min": params["ang_stiff_min"],
        "ang_stiff_max": params["ang_stiff_max"],
        "lin_damping_min": params["lin_damping_min"],
        "lin_damping_max": params["lin_damping_max"],
        "ang_damping_min": params["ang_damping_min"],
        "ang_damping_max": params["ang_damping_max"],
        "dt": params["dt"],
        "inp_bias": params["inp_bias"],
        "anchor_con_fraction": params["anchor_con_fraction"],
        "lin_input_non_zero": params["non_zero_elements"],
        "lin_input_min": params["magnitude_min"],
        "lin_input_max": params["magnitude_max"],
        "ang_input_non_zero": params["non_zero_elements_ang"],
        "ang_input_min": params["magnitude_min_ang"],
        "ang_input_max": params["magnitude_max_ang"],
        "n_connections_fraction": params["n_connections_fraction"],
        "washup_steps": params["washup_steps"],
        "n_connections_ang_fraction": params["n_connections_ang_fraction"],
        "anchor_con_fraction_ang": params["anchor_con_fraction_ang"],
        "n_past_steps_readout": 0,
        "eq_dist_min": params["eq_dist_min"],
        "eq_dist_max": params["eq_dist_max"],
        "eq_dist_min_ang": params["eq_dist_min_ang"],
        "eq_dist_max_ang": params["eq_dist_max_ang"],
    }


def deep_reservoir_config_from_params(params, n_units):
    return {
        "model": "deep_reservoir",
        "n_units": n_units,
        "input_scaling": params["input_scaling"],
        "leaky": params["leaky"],
    }


def config_from_params(params, args):
    if args.model == "deep_reservoir":
        return deep_reservoir_config_from_params(params, args.n_hid)
    return unicycle_config_from_params(params, args.n_hid)


def build_unicycle_model(config, args, device):
    n_units = config["n_units"]
    lin_input_map = make_input_map(
        n_units,
        config["lin_input_non_zero"],
        config["lin_input_min"],
        config["lin_input_max"],
        device,
    )
    ang_input_map = make_input_map(
        n_units,
        config["ang_input_non_zero"],
        config["ang_input_min"],
        config["ang_input_max"],
        device,
    )

    n_connections = int(n_units * config["n_connections_fraction"])
    n_connections_anchor = int(n_units * config["anchor_con_fraction"])
    n_connections_ang = int(n_units * config["n_connections_ang_fraction"])
    n_connections_anchor_ang = int(n_units * config["anchor_con_fraction_ang"])

    model = UnicycleReservoir(
        n_inp=1,
        n_units=n_units,
        dt=config["dt"],
        n_out=args.n_symbols,
        lin_stiff_min=config["lin_stiff_min"],
        lin_stiff_max=config["lin_stiff_max"],
        ang_stiff_min=config["ang_stiff_min"],
        ang_stiff_max=config["ang_stiff_max"],
        lin_damping_min=config["lin_damping_min"],
        lin_damping_max=config["lin_damping_max"],
        ang_damping_min=config["ang_damping_min"],
        ang_damping_max=config["ang_damping_max"],
        eq_dist_min=config["eq_dist_min"],
        eq_dist_max=config["eq_dist_max"],
        eq_dist_min_ang=config["eq_dist_min_ang"],
        eq_dist_max_ang=config["eq_dist_max_ang"],
        n_connections=n_connections,
        inp_bias=config["inp_bias"],
        lin_input_map=lin_input_map,
        n_connections_anchor=n_connections_anchor,
        ang_input_map=ang_input_map,
        n_connections_ang=n_connections_ang,
        n_connections_anchor_ang=n_connections_anchor_ang,
        n_past_steps_readout=config["n_past_steps_readout"],
    ).to(device)
    move_static_tensors(model, device)
    return model


def build_deep_reservoir_model(config, args, device):
    return DeepReservoir(
        1,
        tot_units=config["n_units"],
        spectral_radius=args.rho,
        input_scaling=config["input_scaling"],
        connectivity_recurrent=config["n_units"],
        connectivity_input=config["n_units"],
        leaky=config["leaky"],
    ).to(device)


def build_model(config, args, device):
    if args.model == "deep_reservoir":
        return build_deep_reservoir_model(config, args, device)
    return build_unicycle_model(config, args, device)


def make_run_state(config, device):
    n_units = config["n_units"]
    x, z = torch.rand(2, 1, n_units, device=device)
    theta = torch.rand(1, n_units, device=device) * (4 * math.pi) - (2 * math.pi)
    return x, z, theta, torch.zeros_like(x), torch.zeros_like(x)


@torch.no_grad()
def washup_run_state(model, state, washup_steps, device):
    x, z, theta, s, omega = state
    u = torch.zeros(1, washup_steps, 1, device=device)
    for t in range(washup_steps):
        lin = u[:, t] @ model.lin_input_map
        ang = u[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(lin, ang, x, z, theta, s, omega)
    return tuple(state.detach() for state in (x, z, theta, s, omega))


def set_run_state(model, batch_size, state):
    for name, value in zip(("x_init", "z_init", "theta_init", "s_init", "omega_init"), state):
        setattr(model, name, value.expand(batch_size, -1).clone())


@torch.no_grad()
def features(model, x, args, device, run_state):
    out = []
    for i in range(0, len(x), args.batch_size):
        batch = x[i : i + args.batch_size].to(device)
        if args.model == "deep_reservoir":
            mid_states = model(batch)[1][-1]
        else:
            set_run_state(model, len(batch), run_state)
            mid_states = model(batch, batch)[2]
        if torch.isnan(mid_states).any():
            raise RuntimeError("NaN detected in reservoir states.")
        out.append(mid_states.cpu())
    return torch.cat(out).numpy()


def delay_features(model, delays, args, rng, device, desc, run_state):
    xs, ys = [], []
    for delay in tqdm(delays, desc=desc, leave=False):
        x, y = make_batch(delay, args, rng)
        xs.append(features(model, x, args, device, run_state))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def fit_readout(model, train_delays, args, rng, device, run_state, desc_prefix):
    x_train, y_train = delay_features(
        model,
        train_delays,
        args,
        rng,
        device,
        f"{desc_prefix} train delays",
        run_state,
    )
    if np.isnan(x_train).any():
        raise RuntimeError("NaN detected in training activations.")

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    readout = LogisticRegression(
        C=args.readout_c,
        max_iter=args.readout_max_iter,
    ).fit(x_train, y_train)
    return readout, scaler, float(readout.score(x_train, y_train))


def score_delays(model, readout, scaler, delays, args, rng, device, run_state, desc):
    delay_scores = {}
    for delay in tqdm(delays, desc=desc, leave=False):
        x, y = make_batch(delay, args, rng)
        x = features(model, x, args, device, run_state)
        x = scaler.transform(x)
        delay_scores[delay] = float(readout.score(x, y))
    return delay_scores


def evaluate_config(config, train_delays, score_delay_values, args, rng, device, desc_prefix):
    model = build_model(config, args, device)
    run_state = None
    if args.model == "unicycle":
        run_state = make_run_state(config, device)
    if args.model == "unicycle" and config["washup_steps"] > 0:
        run_state = washup_run_state(model, run_state, config["washup_steps"], device)

    readout, scaler, train_score = fit_readout(
        model, train_delays, args, rng, device, run_state, desc_prefix
    )
    delay_scores = score_delays(
        model,
        readout,
        scaler,
        score_delay_values,
        args,
        rng,
        device,
        run_state,
        f"{desc_prefix} score delays",
    )
    return train_score, delay_scores


def set_torch_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def objective(trial, train_delays, args, device):
    set_torch_seed(args.seed + trial.number, device)
    rng = np.random.default_rng(args.seed)
    config = suggest_config(trial, args)

    try:
        train_score, train_delay_scores = evaluate_config(
            config,
            train_delays,
            train_delays,
            args,
            rng,
            device,
            f"optuna {args.model} {args.n_hid}u trial {trial.number}",
        )
    except (RuntimeError, ValueError) as exc:
        trial.set_user_attr("failure", str(exc))
        return 0.0

    selection_score = float(np.mean(list(train_delay_scores.values())))
    trial.set_user_attr("model", args.model)
    trial.set_user_attr("n_units", args.n_hid)
    trial.set_user_attr("train_score", train_score)
    trial.set_user_attr(
        "selection_train_delay_scores",
        {str(delay): score for delay, score in train_delay_scores.items()},
    )
    return selection_score


def optuna_storage_name(args):
    if args.optuna_storage is not None:
        return args.optuna_storage

    os.makedirs(args.optuna_dir, exist_ok=True)
    db_path = os.path.join(args.optuna_dir, f"{args.optuna_database}.db")
    return f"sqlite:///{db_path}"


def run_optuna_selection(args, train_delays, device):
    storage_name = optuna_storage_name(args)
    study_name = f"{args.model}_{args.n_hid}_units"
    print(f"Starting optimization for model={args.model}, n_units={args.n_hid}")
    study = optuna.create_study(
        storage=storage_name,
        study_name=study_name,
        direction="maximize",
        load_if_exists=args.optuna_load_if_exists,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            train_delays,
            args,
            device,
        ),
        n_trials=args.optuna_trials,
    )
    print(f"Best score for model={args.model}, n_units={args.n_hid}: {study.best_value:.6f}")
    print(f"Best hyperparameters for model={args.model}, n_units={args.n_hid}: {study.best_params}")

    best_config = config_from_params(study.best_params, args)
    return best_config, study, storage_name


def parse_args():
    parser = argparse.ArgumentParser(description="variable-delay copying-memory experiment")
    parser.add_argument(
        "--model",
        choices=("unicycle", "deep_reservoir"),
        default="unicycle",
    )
    parser.add_argument("--resultroot", type=str)
    parser.add_argument("--resultsuffix", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--n_symbols", type=int, default=3)
    parser.add_argument("--train_delays", type=str, default="1,2,5,10,20, 30, 50")
    parser.add_argument(
        "--eval_delays",
        type=str,
        default="1,2,5,10,20,30,50,75,100,150,200",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--distractor_scaling", type=float, default=1.0)
    parser.add_argument("--symbol_scaling", type=float, default=1.0)
    parser.add_argument("--cue_scaling", type=float, default=1.0)
    parser.add_argument("--readout_c", type=float, default=1.0)
    parser.add_argument("--readout_max_iter", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.99, help="Spectral radius for deep reservoir model.")
    parser.add_argument("--n_hid", type=int, required=True)
    parser.add_argument("--optuna_trials", type=int, default=50)
    parser.add_argument("--optuna_database", type=str, default="delayed_copy")
    parser.add_argument("--optuna_dir", type=str, default="optuna_databases")
    parser.add_argument("--optuna_storage", type=str, default=None)
    parser.add_argument(
        "--no_optuna_load_if_exists",
        action="store_false",
        dest="optuna_load_if_exists",
    )
    parser.set_defaults(optuna_load_if_exists=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.resultroot is None:
        warnings.warn("No resultroot provided. Using current location as default.")
        args.resultroot = os.getcwd()
    assert os.path.exists(args.resultroot)
    assert args.n_symbols >= 2 and args.batch_size > 0
    assert args.batch_size >= args.n_symbols
    assert args.n_hid > 0

    train_delays = parse_delays(args.train_delays)
    eval_delays = parse_delays(args.eval_delays)
    args.train_delays = train_delays
    args.eval_delays = eval_delays

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    best_config, best_study, storage_name = run_optuna_selection(args, train_delays, device)
    print(
        f"Selected study {best_study.study_name} from {storage_name} "
        f"with score {best_study.best_value:.6f}"
    )

    train_scores = []
    delay_scores = {delay: [] for delay in eval_delays}
    base_model_seed = args.seed + best_study.best_trial.number

    for trial_idx in range(args.trials):
        rng = np.random.default_rng(args.seed + trial_idx)
        set_torch_seed(base_model_seed + trial_idx, device)
        train_score, scores = evaluate_config(
            best_config,
            train_delays,
            eval_delays,
            args,
            rng,
            device,
            f"selected trial {trial_idx + 1}",
        )
        train_scores.append(train_score)
        for delay, score in scores.items():
            delay_scores[delay].append(score)

    delay_mean = {d: float(np.mean(v)) for d, v in delay_scores.items()}
    delay_std = {d: float(np.std(v)) for d, v in delay_scores.items()}

    print("delay,mean_accuracy,std_accuracy,beyond_train_range")
    for delay in eval_delays:
        print(f"{delay},{delay_mean[delay]:.6f},{delay_std[delay]:.6f},{delay > max(train_delays)}")

    log_name = f"DelayedCopy_log_{args.model}{args.resultsuffix}.txt"
    log = "".join(f"{k}: {v}, " for k, v in vars(args).items())
    log += (
        f"optuna_storage: {storage_name} "
        f"optuna_best_study: {best_study.study_name} "
        f"optuna_best_value: {best_study.best_value} "
        f"optuna_best_params: {best_study.best_params} "
        f"selected_config: {best_config} "
        f"train_readout: {[str(round(x, 4)) for x in train_scores]} "
        f"mean/std train_readout: {np.mean(train_scores), np.std(train_scores)} "
        f"delay_mean: {delay_mean} delay_std: {delay_std}"
    )
    with open(os.path.join(args.resultroot, log_name), "a") as f:
        f.write(log + "\n")


if __name__ == "__main__":
    main()

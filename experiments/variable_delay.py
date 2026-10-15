import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __package__:
    from .delayed_copy import (
        build_model,
        config_from_params,
        features,
        make_distractor,
        make_run_state,
        optuna_storage_name,
        parse_delays,
        set_torch_seed,
        suggest_config,
        washup_run_state,
    )
else:
    from experiments.delayed_copy import (
        build_model,
        config_from_params,
        features,
        make_distractor,
        make_run_state,
        optuna_storage_name,
        parse_delays,
        set_torch_seed,
        suggest_config,
        washup_run_state,
    )


SAMPLE_PROBE_PAIRS = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.int64,
)


def stimulus_values(args):
    return np.array(
        [-args.stimulus_scaling, args.stimulus_scaling],
        dtype=np.float32,
    )


def make_batch(delay, args, rng):
    x = np.zeros((args.batch_size, delay + 2, 1), dtype=np.float32)
    y = np.zeros(args.batch_size, dtype=np.int64)
    values = stimulus_values(args)

    for start in range(0, args.batch_size, len(SAMPLE_PROBE_PAIRS)):
        stop = min(start + len(SAMPLE_PROBE_PAIRS), args.batch_size)
        pairs = SAMPLE_PROBE_PAIRS[: stop - start]
        samples = pairs[:, 0]
        probes = pairs[:, 1]

        x[start:stop, 0, 0] = values[samples]
        x[start:stop, 1 : delay + 1, 0] = make_distractor(delay, args, rng)
        x[start:stop, delay + 1, 0] = values[probes]
        y[start:stop] = samples == probes
    x = torch.from_numpy(x)

    return x, y


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
        model,
        train_delays,
        args,
        rng,
        device,
        run_state,
        desc_prefix,
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
    trial.set_user_attr("task", "variable_delay_match_to_sample")
    trial.set_user_attr("model", args.model)
    trial.set_user_attr("n_units", args.n_hid)
    trial.set_user_attr("train_score", train_score)
    trial.set_user_attr(
        "selection_train_delay_scores",
        {str(delay): score for delay, score in train_delay_scores.items()},
    )
    return selection_score


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
    parser = argparse.ArgumentParser(
        description="variable-delay match-to-sample experiment"
    )
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
    parser.add_argument("--train_delays", type=str, default="1,2,5,10,20,30,50")
    parser.add_argument(
        "--eval_delays",
        type=str,
        default="1,2,5,10,20,30,50,75,100,150,200",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--distractor_scaling", type=float, default=0.01)
    parser.add_argument("--stimulus_scaling", type=float, default=1.0)
    parser.add_argument("--readout_c", type=float, default=1.0)
    parser.add_argument("--readout_max_iter", type=int, default=1000)
    parser.add_argument(
        "--rho",
        type=float,
        default=0.99,
        help="Spectral radius for deep reservoir model.",
    )
    parser.add_argument("--n_hid", type=int, required=True)
    parser.add_argument("--optuna_trials", type=int, default=50)
    parser.add_argument("--optuna_database", type=str, default="variable_delay")
    parser.add_argument("--optuna_dir", type=str, default="optuna_databases")
    parser.add_argument("--optuna_storage", type=str, default=None)
    parser.add_argument(
        "--no_optuna_load_if_exists",
        action="store_false",
        dest="optuna_load_if_exists",
    )
    parser.set_defaults(optuna_load_if_exists=True)

    args = parser.parse_args()
    args.n_symbols = 2
    return args


def main():
    args = parse_args()

    if args.resultroot is None:
        warnings.warn("No resultroot provided. Using current location as default.")
        args.resultroot = os.getcwd()
    assert os.path.exists(args.resultroot)
    assert args.batch_size >= len(SAMPLE_PROBE_PAIRS)
    assert args.n_hid > 0
    assert args.stimulus_scaling != 0

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

    log_name = f"VariableDelay_log_{args.model}{args.resultsuffix}.txt"
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

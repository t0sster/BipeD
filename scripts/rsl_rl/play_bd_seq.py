# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint with sequential command parameters."""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL agent with sequential command parameters.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--eval_steps", type=int, default=1200, help="Steps per candidate evaluation.")
parser.add_argument("--eval_time_s", type=float, default=5.0, help="Sim time per candidate; overrides eval_steps.")
parser.add_argument("--population", type=int, default=16, help="Population size.")
parser.add_argument("--elite", type=int, default=4, help="Elite count per generation.")
parser.add_argument("--generations", type=int, default=8, help="Number of generations.")
parser.add_argument("--sigma", type=float, default=0.15, help="Mutation sigma as a fraction of range.")
parser.add_argument("--sigma_decay", type=float, default=0.9, help="Per-generation sigma decay factor.")
parser.add_argument("--best_parent_prob", type=float, default=0.7, help="Chance to mutate best candidate.")
parser.add_argument("--step_size", type=float, default=0.05, help="Discretization step for command params.")
parser.add_argument("--batch_eval", action="store_true", default=False, help="Evaluate population in parallel.")
parser.add_argument("--tracking_weight", type=float, default=1.0, help="Tracking error weight.")
parser.add_argument("--cot_weight", type=float, default=1.0, help="Cost of transport weight.")
parser.add_argument("--param_sets", type=str, default=None, help="Path to JSON list of parameter sets.")
parser.add_argument("--output", type=str, default=None, help="Directory to save logs and plots.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import BipeD.tasks  # noqa: F401

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class CommandParamSpace:
    def __init__(self, env, step_size):
        self._env = env
        self._terms = env.command_manager
        self._params = []
        self._bounds = []
        self._step_size = step_size

        self._add_step_ranges()
        self._add_base_height_ranges()
        self._add_gait_ranges()
        self._add_velocity_ranges()

    @property
    def names(self):
        return [name for name, _ in self._params]

    @property
    def bounds(self):
        return list(self._bounds)

    def _add_param(self, name, getter, min_val, max_val):
        self._params.append((name, getter))
        self._bounds.append((float(min_val), float(max_val)))

    def _add_step_ranges(self):
        term = self._terms.get_term("lip_step_command")
        ranges = term.cfg.ranges
        if ranges is None:
            return
        if ranges.step_length is not None:
            self._add_param("step_length", lambda: term, *ranges.step_length)
        if ranges.step_width is not None:
            self._add_param("step_width", lambda: term, *ranges.step_width)
        if ranges.step_period_s is not None:
            self._add_param("step_period_s", lambda: term, *ranges.step_period_s)

    def _add_base_height_ranges(self):
        term = self._terms.get_term("base_height_command")
        ranges = term.cfg.ranges
        self._add_param("base_height", lambda: term, *ranges.height)

    def _add_gait_ranges(self):
        term = self._terms.get_term("gait_command")
        ranges = term.cfg.ranges
        self._add_param("gait_frequency", lambda: term, *ranges.frequencies)
        self._add_param("gait_offset", lambda: term, *ranges.offsets)
        self._add_param("gait_duration", lambda: term, *ranges.durations)

    def _add_velocity_ranges(self):
        term = self._terms.get_term("base_velocity")
        ranges = term.cfg.ranges
        min_vx, max_vx = ranges.lin_vel_x
        self._add_param("lin_vel_x", lambda: term, max(0.0, min_vx), max_vx)

    def sample_uniform(self, count, rng):
        samples = []
        for _ in range(count):
            values = {}
            for (name, _), (low, high) in zip(self._params, self._bounds, strict=True):
                values[name] = _quantize(rng.uniform(low, high), low, high, self._step_size)
            samples.append(values)
        return samples


class CommandApplier:
    def __init__(self, env, space):
        self._env = env
        self._terms = env.command_manager
        self._space = space

    def apply(self, params):
        env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._apply_params(params, env_ids)

    def apply_single(self, params, env_id):
        env_ids = torch.tensor([env_id], device=self._env.device)
        self._apply_params(params, env_ids)

    def apply_batch(self, params_list):
        if len(params_list) > self._env.num_envs:
            raise ValueError("params_list exceeds available environments")
        for env_id, params in enumerate(params_list):
            self.apply_single(params, env_id)

    def _apply_params(self, params, env_ids):
        params = self._quantize_params(params)

        if "step_length" in params or "step_width" in params or "step_period_s" in params:
            term = self._terms.get_term("lip_step_command")
            ranges = term.cfg.ranges
            if "step_length" in params:
                ranges.step_length = (params["step_length"], params["step_length"])
            if "step_width" in params:
                ranges.step_width = (params["step_width"], params["step_width"])
            if "step_period_s" in params:
                ranges.step_period_s = (params["step_period_s"], params["step_period_s"])
            term._resample_command(env_ids)

        if "base_height" in params:
            term = self._terms.get_term("base_height_command")
            term.cfg.ranges.height = (params["base_height"], params["base_height"])
            term._resample_command(env_ids)

        if any(k in params for k in ("gait_frequency", "gait_offset", "gait_duration")):
            term = self._terms.get_term("gait_command")
            ranges = term.cfg.ranges
            if "gait_frequency" in params:
                ranges.frequencies = (params["gait_frequency"], params["gait_frequency"])
            if "gait_offset" in params:
                ranges.offsets = (params["gait_offset"], params["gait_offset"])
            if "gait_duration" in params:
                ranges.durations = (params["gait_duration"], params["gait_duration"])
            term._resample_command(env_ids)

        if any(k in params for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")):
            term = self._terms.get_term("base_velocity")
            ranges = term.cfg.ranges
            if "lin_vel_x" in params:
                vx = max(0.05, params["lin_vel_x"])
                ranges.lin_vel_x = (0.05, vx)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
            if hasattr(ranges, "heading"):
                ranges.heading = (0.0, 0.0)
            term._resample_command(env_ids)

    def _quantize_params(self, params):
        bounds = {name: bound for (name, _), bound in zip(self._space._params, self._space._bounds, strict=True)}
        quantized = {}
        for key, val in params.items():
            if key in bounds:
                low, high = bounds[key]
                quantized[key] = _quantize(val, low, high, self._space._step_size)
            else:
                quantized[key] = val
        return quantized


class GaitEvaluator:
    def __init__(self, env, policy, reset_fn, tracking_weight, cot_weight):
        self._env = env
        self._policy = policy
        self._reset_fn = reset_fn
        self._tracking_weight = tracking_weight
        self._cot_weight = cot_weight

    def evaluate(self, steps):
        env = self._env
        env.reset()
        obs = env.get_observations()

        base = env.unwrapped.scene["robot"]
        mass = base.data.default_mass[0].sum().item()
        dt = env.unwrapped.step_dt
        start_pos = base.data.root_pos_w[0, :2].clone()

        energy = 0.0
        tracking_err = 0.0

        for _ in range(steps):
            with torch.no_grad():
                actions = self._policy(obs)
                obs, _, dones, _ = env.step(actions)

                if self._reset_fn is not None:
                    self._reset_fn(dones)

                torque = base.data.applied_torque[0]
                joint_vel = base.data.joint_vel[0]
                power = torch.sum(torch.abs(torque * joint_vel)).item()
                energy += power * dt

                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0]
                cmd_vx, cmd_vy, cmd_wz = cmd[0].item(), cmd[1].item(), cmd[2].item()

                vel = base.data.root_lin_vel_w[0]
                ang = base.data.root_ang_vel_w[0]
                vel_err = (vel[0].item() - cmd_vx) ** 2 + (vel[1].item() - cmd_vy) ** 2
                ang_err = (ang[2].item() - cmd_wz) ** 2

                height_cmd = env.unwrapped.command_manager.get_command("base_height_command")[0, 0].item()
                height_err = (base.data.root_pos_w[0, 2].item() - height_cmd) ** 2

                tracking_err += vel_err + ang_err + height_err

        end_pos = base.data.root_pos_w[0, :2].clone()
        distance = torch.norm(end_pos - start_pos).item()
        distance = max(distance, 1e-4)

        cot = energy / (mass * distance)
        tracking = tracking_err / max(steps, 1)
        return {
            "cost_of_transport": cot,
            "tracking_error": tracking,
        }

    def evaluate_batch(self, steps, count):
        env = self._env
        env.reset()
        obs = env.get_observations()

        base = env.unwrapped.scene["robot"]
        device = env.unwrapped.device
        mass = base.data.default_mass.to(device).sum(dim=1)
        dt = env.unwrapped.step_dt
        start_pos = base.data.root_pos_w[:, :2].clone()

        energy = torch.zeros(env.num_envs, device=device)
        tracking_err = torch.zeros(env.num_envs, device=device)

        for _ in range(steps):
            with torch.no_grad():
                actions = self._policy(obs)
                obs, _, dones, _ = env.step(actions)

                if self._reset_fn is not None:
                    self._reset_fn(dones)

                torque = base.data.applied_torque
                joint_vel = base.data.joint_vel
                power = torch.sum(torch.abs(torque * joint_vel), dim=1)
                energy += power * dt

                cmd = env.unwrapped.command_manager.get_command("base_velocity")
                vel = base.data.root_lin_vel_w
                ang = base.data.root_ang_vel_w
                vel_err = torch.square(vel[:, 0] - cmd[:, 0]) + torch.square(vel[:, 1] - cmd[:, 1])
                ang_err = torch.square(ang[:, 2] - cmd[:, 2])

                height_cmd = env.unwrapped.command_manager.get_command("base_height_command")[:, 0]
                height_err = torch.square(base.data.root_pos_w[:, 2] - height_cmd)

                tracking_err += vel_err + ang_err + height_err

        end_pos = base.data.root_pos_w[:, :2].clone()
        distance = torch.norm(end_pos - start_pos, dim=1).clamp(min=1e-4)

        cot = energy / (mass * distance)
        tracking = tracking_err / max(steps, 1)
        results = []
        for idx in range(count):
            results.append(
                {
                    "cost_of_transport": float(cot[idx].item()),
                    "tracking_error": float(tracking[idx].item()),
                }
            )
        return results


class EvolutionaryOptimizer:
    def __init__(
        self,
        space,
        rng,
        population,
        elite,
        sigma,
        sigma_decay,
        best_parent_prob,
        cot_weight,
        tracking_weight,
        log_every=1,
    ):
        self._space = space
        self._rng = rng
        self._population = population
        self._elite = elite
        self._sigma = sigma
        self._sigma_decay = max(0.0, min(1.0, sigma_decay))
        self._best_parent_prob = max(0.0, min(1.0, best_parent_prob))
        self._cot_weight = cot_weight
        self._tracking_weight = tracking_weight
        self._log_every = max(1, log_every)

    def optimize(self, generations, evaluate_fn, batch_eval=False):
        candidates = self._space.sample_uniform(self._population, self._rng)
        results = []
        history = []

        for gen_idx in range(generations):
            scored = []
            if batch_eval:
                metrics_list = evaluate_fn(candidates)
                for cand_idx, (params, metrics) in enumerate(zip(candidates, metrics_list, strict=True)):
                    record = {"params": params, "metrics": metrics, "generation": gen_idx, "candidate": cand_idx}
                    scored.append(record)
                    history.append(record)
            else:
                for cand_idx, params in enumerate(candidates):
                    metrics = evaluate_fn(params)
                    record = {"params": params, "metrics": metrics, "generation": gen_idx, "candidate": cand_idx}
                    scored.append(record)
                    history.append(record)

            _normalize_scores(scored, self._cot_weight, self._tracking_weight)
            scored.sort(key=lambda item: item["metrics"]["score"], reverse=True)
            if gen_idx % self._log_every == 0:
                best = scored[0]
                avg_score = sum(item["metrics"]["score"] for item in scored) / max(1, len(scored))
                print(
                    "[INFO] Gen {}/{} | best score {:.6f} | avg score {:.6f} | cot {:.6f} | tracking {:.6f}".format(
                        gen_idx + 1,
                        generations,
                        best["metrics"]["score"],
                        avg_score,
                        best["metrics"]["cost_of_transport"],
                        best["metrics"]["tracking_error"],
                    )
                )
            results.extend(scored)

            elites = scored[: self._elite]
            candidates = [entry["params"] for entry in elites]
            current_sigma = self._sigma * (self._sigma_decay ** gen_idx)

            while len(candidates) < self._population:
                if self._rng.random() < self._best_parent_prob:
                    parent = elites[0]["params"]
                else:
                    parent = self._rng.choice(elites)["params"]
                child = self._mutate(parent, current_sigma)
                candidates.append(child)

        best = max(results, key=lambda item: item["metrics"]["score"])
        return best, results, history

    def _mutate(self, parent, sigma):
        child = {}
        for (name, _), (low, high) in zip(self._space._params, self._space._bounds, strict=True):
            span = high - low
            value = parent[name] + self._rng.gauss(0.0, sigma * span)
            value = max(low, min(high, value))
            child[name] = _quantize(value, low, high, self._space._step_size)
        return child


def _load_param_sets(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("param_sets must be a list of dictionaries")
    return data


def _normalize_scores(items, cot_weight, tracking_weight, eps=1e-6):
    if not items:
        return
    cot_mean = sum(item["metrics"]["cost_of_transport"] for item in items) / len(items)
    tracking_mean = sum(item["metrics"]["tracking_error"] for item in items) / len(items)
    cot_mean = max(cot_mean, eps)
    tracking_mean = max(tracking_mean, eps)

    for item in items:
        cot_norm = item["metrics"]["cost_of_transport"] / cot_mean
        tracking_norm = item["metrics"]["tracking_error"] / tracking_mean
        objective = cot_weight * cot_norm + tracking_weight * tracking_norm
        item["metrics"]["cot_norm"] = cot_norm
        item["metrics"]["tracking_norm"] = tracking_norm
        item["metrics"]["score"] = 1.0 / (eps + objective)


def _quantize(value, low, high, step):
    if step <= 0.0:
        return float(max(low, min(high, value)))
    snapped = round((value - low) / step) * step + low
    snapped = max(low, min(high, snapped))
    return float(snapped)


class ResultLogger:
    def __init__(self, log_dir):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename, payload):
        path = self._log_dir / filename
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def write_csv(self, filename, rows):
        if not rows:
            return
        path = self._log_dir / filename
        keys = sorted(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def plot_metrics(self, history):
        if plt is None or not history:
            return
        steps = list(range(len(history)))
        metrics = {
            "score": [item["metrics"]["score"] for item in history],
            "cost_of_transport": [item["metrics"]["cost_of_transport"] for item in history],
            "tracking_error": [item["metrics"]["tracking_error"] for item in history],
        }

        for name, series in metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(steps, series, label=name)
            plt.xlabel("evaluation")
            plt.ylabel(name)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self._log_dir / f"{name}.png")
            plt.close()

        best_by_gen = {}
        for item in history:
            gen = item.get("generation", 0)
            best = best_by_gen.get(gen)
            if best is None or item["metrics"]["score"] > best["metrics"]["score"]:
                best_by_gen[gen] = item

        gens = sorted(best_by_gen.keys())
        best_metrics = {
            "score": [best_by_gen[g]["metrics"]["score"] for g in gens],
            "cost_of_transport": [best_by_gen[g]["metrics"]["cost_of_transport"] for g in gens],
            "tracking_error": [best_by_gen[g]["metrics"]["tracking_error"] for g in gens],
        }

        for name, series in best_metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(gens, series, label=f"best_{name}")
            plt.xlabel("generation")
            plt.ylabel(f"best_{name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self._log_dir / f"best_{name}.png")
            plt.close()


def main():
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "").replace("-SeqPlay", "")

    if args_cli.batch_eval:
        if args_cli.num_envs is None:
            env_cfg.scene.num_envs = args_cli.population
        elif args_cli.num_envs < args_cli.population:
            raise ValueError("num_envs must be >= population for batch_eval")
        else:
            env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_seq"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.eval_time_s is not None:
        eval_steps = max(1, int(args_cli.eval_time_s / env.unwrapped.step_dt))
    else:
        eval_steps = args_cli.eval_steps

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    reset_fn = None
    if version.parse(installed_version) >= version.parse("4.0.0"):
        reset_fn = policy.reset
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic
        reset_fn = policy_nn.reset

    space = CommandParamSpace(env.unwrapped, args_cli.step_size)
    applier = CommandApplier(env.unwrapped, space)
    evaluator = GaitEvaluator(env, policy, reset_fn, args_cli.tracking_weight, args_cli.cot_weight)

    rng = random.Random(agent_cfg.seed)

    project_root = Path(__file__).resolve().parents[2]
    base_log_dir = Path(args_cli.output) if args_cli.output else project_root / "logs" / "optimization"
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = ResultLogger(base_log_dir / run_id)

    if args_cli.param_sets:
        param_sets = _load_param_sets(args_cli.param_sets)
        results = []
        history = []
        for params in param_sets:
            applier.apply(params)
            metrics = evaluator.evaluate(eval_steps)
            record = {"params": params, "metrics": metrics, "generation": 0, "candidate": len(history)}
            results.append(record)
            history.append(record)
        _normalize_scores(results, args_cli.cot_weight, args_cli.tracking_weight)
        best = max(results, key=lambda item: item["metrics"]["score"]) if results else None
    else:
        optimizer = EvolutionaryOptimizer(
            space,
            rng,
            args_cli.population,
            args_cli.elite,
            args_cli.sigma,
            args_cli.sigma_decay,
            args_cli.best_parent_prob,
            args_cli.cot_weight,
            args_cli.tracking_weight,
        )

        if args_cli.batch_eval:
            def _evaluate_batch(candidates):
                applier.apply_batch(candidates)
                return evaluator.evaluate_batch(eval_steps, len(candidates))

            best, results, history = optimizer.optimize(args_cli.generations, _evaluate_batch, batch_eval=True)
        else:
            def _evaluate(params):
                applier.apply(params)
                return evaluator.evaluate(eval_steps)

            best, results, history = optimizer.optimize(args_cli.generations, _evaluate)

    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "best": best,
        "results": results,
    }

    logger.write_json("summary.json", summary)
    logger.write_json("history.json", history)
    flat_rows = []
    for item in history:
        row = {"generation": item.get("generation", 0), "candidate": item.get("candidate", 0)}
        row.update(item["params"])
        row.update(item["metrics"])
        flat_rows.append(row)
    logger.write_csv("history.csv", flat_rows)
    logger.plot_metrics(history)

    if best is not None:
        print("[INFO] Best candidate:")
        print(json.dumps(best, indent=2))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

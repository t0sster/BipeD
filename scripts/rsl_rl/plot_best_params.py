import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _load_history(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _best_by_generation(history):
    best = {}
    for item in history:
        gen = item.get("generation", 0)
        current = best.get(gen)
        if current is None or item["metrics"]["score"] > current["metrics"]["score"]:
            best[gen] = item
    return best


def _collect_params(best_by_gen):
    gens = sorted(best_by_gen.keys())
    param_names = []
    if gens:
        param_names = sorted(best_by_gen[gens[0]]["params"].keys())
    series = {name: [] for name in param_names}
    for gen in gens:
        params = best_by_gen[gen]["params"]
        for name in param_names:
            series[name].append(params.get(name))
    return gens, series


def _plot_series(output_dir, gens, series):
    if plt is None:
        raise RuntimeError("matplotlib is required to plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, values in series.items():
        plt.figure(figsize=(10, 6))
        plt.plot(gens, values, label=name)
        plt.xlabel("generation")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"best_{name}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot best-by-generation parameter curves.")
    parser.add_argument("--history", required=True, help="Path to history.json")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    args = parser.parse_args()

    history = _load_history(args.history)
    best_by_gen = _best_by_generation(history)
    gens, series = _collect_params(best_by_gen)
    _plot_series(Path(args.output), gens, series)


if __name__ == "__main__":
    main()

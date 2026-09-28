"""ReplaceMe pipeline: for each n, select the cut, fit and fuse T, delete the
blocks, then evaluate utility and safety."""
import torch

import config
from common import core
from replaceme.method import check_fusion, fit_transform, select_cut

ALPHA = 0.0   # ridge strength; 0.0 = ordinary least squares


def build_arm(n, tok, cut_loader, full_loader):
    arm = f"replaceme_n{n}"
    model = core.load_model()   # fresh copy per n, since surgery is in place
    L = model.config.num_hidden_layers
    n_params_base = core.count_params(model)

    i_star, dist = select_cut(model, cut_loader, n)
    removed = list(range(i_star + 1, i_star + n + 1))

    T = fit_transform(model, full_loader, i_star, n, ALPHA)
    torch.save(T, config.MODELS_DIR / f"T_{arm}.pt")

    check_fusion(model, i_star, T, next(iter(cut_loader)))   # also fuses T
    core.delete_layers(model, removed)
    core.save_arm(model, tok, config.MODELS_DIR / arm)

    n_params = core.count_params(model)
    core.write_json(config.RESULTS_DIR / f"build_{arm}.json", {
        "n": n, "i_star": i_star, "removed_layers": removed,
        "angular_distance": float(dist[i_star]), "distance_curve": dist.tolist(),
        "layers_after": model.config.num_hidden_layers,
        "compression_ratio": 1 - n_params / n_params_base})
    del model, T
    core.free_gpu()
    return arm


def main():
    core.setup()
    tok = core.load_tokenizer()

    calib = core.load_calibration(tok, config.N_CALIB)
    cut_loader = core.make_loader(calib.select(range(config.N_CUT)), tok)
    full_loader = core.make_loader(calib, tok)
    arms = [build_arm(n, tok, cut_loader, full_loader) for n in config.N_VALUES]

    benchmarks = core.load_safety_benchmarks()
    for arm in arms:
        model_dir = str(config.MODELS_DIR / arm)
        core.run_utility_eval(arm, model_dir)
        core.generate_safety(arm, model_dir, tok, benchmarks)


if __name__ == "__main__":
    main()

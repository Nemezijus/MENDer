from engine.extras.allen_institute.datafetch.visual_behavior import load_visual_behavior_experiment
from engine.extras.allen_institute.preprocessing.visual_behavior.natural_images import build_natural_image_trials
from engine.extras.allen_institute.preprocessing.common import save_trial_dataset_npz
from instances.logreg_classify_with_shuffle import run_logreg_decoding
from instances.ridge_shuffle_decoding import run_ridge_decoding

from engine.extras.allen_institute.datafetch.visual_coding import load_vc_experiment
from engine.extras.allen_institute.preprocessing.visual_coding.drifting_gratings import build_vc_drifting_gratings_trials
from visualizations.allen_institute.allen_vc_plots import (
    plot_orientation_distribution,
    plot_population_means,
    plot_example_tuning_curves,
)



def example_run(
    x_path="data/test/Fdff.mat",
    y_path="data/test/rotationalVelocity.mat",
):
    run_ridge_decoding(x_path, y_path)


def allen_example_run():
    # 1) Fetch experiment
    exp = load_visual_behavior_experiment(951980471, cache_dir="data/allen_cache")

    # 2) Build trial-level dataset (natural images)
    out = build_natural_image_trials(exp, feature="mean", pre_window_s=0.5, post_window_s=1.0)
    X, y = out["X"], out["y"]

    # 3) Save NPZ so the instance script can load it
    npz_path = "data/allen/vb_exp_951980471_image_name_mean.npz"
    save_trial_dataset_npz(out, npz_path)   # or: np.savez_compressed(npz_path, X=X, y=y)

    # 4) Run your logistic regression with shuffle-baseline
    real_score, shuffled_scores = run_logreg_decoding(
        npz_path=npz_path,      # <- use NPZ route
        n_shuffles=200,
        train_frac=0.8,
        standardize=True,       # recommended
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        metric="accuracy",
        rng=42,
    )
    print("Done. Accuracy:", real_score)


def allen_vc_example_run():
    ds = load_vc_experiment(512326618)
    out = build_vc_drifting_gratings_trials(
        ds,
        feature="mean",
        label_by="orientation",
        bin_orientations=True,   # try this to make discrete classes
        bin_size_deg=45.0
    )
    npz_path = "data/allen_vc/vc512326618_dg_orientation_mean.npz"
    save_trial_dataset_npz(out, npz_path)
    plot_orientation_distribution(out)
    plot_population_means(out, topk=16)
    plot_example_tuning_curves(out, n_examples=6)

    run_logreg_decoding(
        npz_path=npz_path,
        train_frac=0.8,
        standardize=True,
        n_shuffles=200,
        metric="accuracy",
        rng=42,
    )
if __name__ == "__main__":
    allen_vc_example_run()
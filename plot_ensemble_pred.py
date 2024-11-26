import pickle

import numpy as np
from metrics import NMAE, NMSE, r2_score, MAE, log_likelihood_score, _identity
from plot import (
    plot_num_members_metrics,
    plot_rollout_metrics,
    plot_single_field,
    plot_uncertainty_field,
    plot_double_rollout_metrics,
    plot_double_rollout_ep
)

# load data
data_dir = "/Users/yixuan.sun/Downloads/hpo_uq_pred_masked/"
criterion = "topK"
data = np.load(data_dir + f"{criterion}_test_ensemble_predictions.npz")

true_labels = data["true_labels"]
pred_mean = data["pred_mean"]
aleatoric_uc = data["aleatoric_uc"]
epistemic_uc = data["epistemic_uc"]

sample_idx = 0
var_id = 2
var_names = [
    "Salinity",
    "Temperature",
    "Meridional Velocity",
    "Zonal Velocity",
    "GM",
]

added_var = [
    0.0001,
    0.25,
    0.0025,
    0.0025,
]


with open(data_dir + "SOMA_mask.pkl", "rb") as f:
    mask = pickle.load(f)
mask1 = mask["mask1"]
mask2 = mask["mask2"]
mask = np.logical_or(mask1, mask2)[0, 0, :, :, 0]

# compute all the metrics for given state variable
for var_id_ in range(4):
    print(var_id, ":", var_names[var_id_])
    print(
        "state var range",
        true_labels[sample_idx, var_id_][~mask].min(),
        true_labels[sample_idx, var_id_][~mask].max(),
    )
    print(
        "pred range",
        pred_mean[sample_idx, var_id_][~mask].min(),
        pred_mean[sample_idx, var_id_][~mask].max(),
    )
    print(
        "NMSE: %.4e "
        % NMSE(
            true_labels[sample_idx, var_id_][~mask],
            pred_mean[sample_idx, var_id_][~mask],
        ),
    )
    print(
        "NMAE: %.4e "
        % NMAE(
            true_labels[sample_idx, var_id_][~mask],
            pred_mean[sample_idx, var_id_][~mask],
        ),
    )
    print(
        "r2_score: %.5f"
        % r2_score(
            true_labels[sample_idx, var_id_][~mask],
            pred_mean[sample_idx, var_id_][~mask],
        ),
    )

    # print the aleatoric uncertainty error
    print(
        "aleatoric_uc: %.4e" % aleatoric_uc[sample_idx, var_id_][~mask].mean(),
    )
    print("aleatoric_uc NMAE: %.4e" % NMAE(aleatoric_uc[sample_idx, var_id_][~mask], added_var[var_id_]))

    # loglikelihood
    print(
        "log_likelihood: %.4e"
        % log_likelihood_score(
            pred_mean[sample_idx, var_id_][~mask],
            aleatoric_uc[sample_idx, var_id_][~mask],
            true_labels[sample_idx, var_id_][~mask],
        ),
    )


# assert False

    # plot true field
    print(true_labels.shape)
    vmin = true_labels[sample_idx, var_id_][~mask].min()
    vmax = true_labels[sample_idx, var_id_][~mask].max()
    fig = plot_single_field(
        field=true_labels[sample_idx, var_id_],
        field_name=var_names[var_id_],
        vmin=vmin,
        vmax=vmax,
        mask=mask,
    )
    fig.savefig(
        f"./figs/ensemble_pred/true_field_{var_names[var_id_]}_{criterion}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    # plot mean prediction
    fig = plot_single_field(
        field=pred_mean[sample_idx, var_id_],
        field_name=var_names[var_id_],
        vmin=vmin,
        vmax=vmax,
        mask=mask,
    )
    fig.savefig(
        f"./figs/ensemble_pred/pred_mean_{var_names[var_id_]}_{criterion}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    # plot the difference true - pred
    exp_diff = np.mean(true_labels[sample_idx, var_id_] * 0.1)
    fig = plot_single_field(
        field=true_labels[sample_idx, var_id_] - pred_mean[sample_idx, var_id_],
        field_name=var_names[var_id_],
        vmin=-exp_diff,
        vmax=exp_diff,
        mask=mask,
    )
    fig.savefig(
        f"./figs/ensemble_pred/diff_{var_names[var_id_]}_{criterion}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    # plot aleatoric uncertainty
    fig = plot_uncertainty_field(
        field=aleatoric_uc[sample_idx, var_id_],
        field_name=var_names[var_id_],
        vmin=added_var[var_id_] * 0.9,
        vmax=added_var[var_id_] * 1.1,
        mask=mask,
    )
    fig.savefig(
        f"./figs/ensemble_pred/aleatoric_uc_{var_names[var_id_]}_{criterion}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    # plot aleatoric uncertainty true - pred

    exp_diff_al = added_var[var_id_] * 0.1
    fig = plot_uncertainty_field(
        field=added_var[var_id_] - aleatoric_uc[sample_idx, var_id_],
        field_name=var_names[var_id_],
        vmin=-exp_diff_al,
        vmax=exp_diff_al,
        mask=mask,
    )
    fig.savefig(
        f"./figs/ensemble_pred/aleatoric_uc_diff_{var_names[var_id_]}_{criterion}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    # plot epistemic uncertainty
    fig = plot_uncertainty_field(
        field=epistemic_uc[sample_idx, var_id_],
        field_name=var_names[var_id_],
        vmin=None,
        vmax=None,
        mask=mask,
    )
    fig.savefig(
        f"./figs/ensemble_pred/epistemic_uc_{var_names[var_id_]}_{criterion}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

#-----------------end for--------

# plot rollout
rollouts = np.load(data_dir + f"{criterion}_test_rollout.npz")

if criterion == "topK":
    rollouts2 = np.load(data_dir + "greedy_test_rollout.npz")
else:
    rollouts2 = np.load(data_dir + "topK_test_rollout.npz")

true = rollouts["true"]
pred = rollouts["pred"]
epistemic_uc = rollouts["ep_uc"]

true2 = rollouts2["true"]
pred2 = rollouts2["pred"]
epistemic_uc2 = rollouts2["ep_uc"]


fig = plot_rollout_metrics(true, pred, mask, score_fn=NMAE)
fig.savefig(
    f"./figs/ensemble_pred/{criterion}_rollout_NMAE.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig = plot_rollout_metrics(true, pred, mask, score_fn=r2_score)
fig.savefig(
    f"./figs/ensemble_pred/{criterion}_rollout_r2_score.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig = plot_rollout_metrics(epistemic_uc, epistemic_uc, mask, score_fn=_identity)
fig.savefig(
    f"./figs/ensemble_pred/{criterion}_rollout_epistemic_uc.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig = plot_double_rollout_metrics(true, pred, pred2, mask, score_fn=NMAE)
fig.savefig(
    f"./figs/ensemble_pred/{criterion}_rollout_double_NMAE.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig = plot_double_rollout_ep(epistemic_uc, epistemic_uc2, mask)
fig.savefig(
    f"./figs/ensemble_pred/{criterion}_rollout_double_epistemic_uc.pdf",
    format="pdf",
    bbox_inches="tight",
)


# plot ensemble members vs scores
scores_topk = np.load(data_dir + "topK_val_num_members_scores.npy")
scores_greedy = np.load(data_dir + "greedy_val_num_members_scores.npy")


# fig = plot_num_members_metrics((scores_topk, scores_greedy))
# fig.savefig(
#     f"./figs/ensemble_pred/{criterion}num_members_val.pdf",
#     format="pdf",
#     bbox_inches="tight",
# )

# plot ensemble members vs scores on the test set
scores_topk_test = np.load(data_dir + "topK_test_num_members_scores.npy")
scores_greedy_test = np.load(data_dir + "greedy_test_num_members_scores.npy")


fig = plot_num_members_metrics(
    (scores_topk, scores_greedy), (scores_topk_test, scores_greedy_test)
)
fig.savefig(
    f"./figs/ensemble_pred/{criterion}num_members_test.pdf",
    format="pdf",
    bbox_inches="tight",
)

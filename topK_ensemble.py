import numpy as np
import os
from data import SOMAdata
from ensemble_predict import EnsemblePredictor
from metrics import MSE, log_likelihood_score
from torch.utils.data import DataLoader


def main():
    valset = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "val",
        y_noise=True,
        transform=True,
    )
    testSet = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "test",
        transform=True,
    )

    predictor = EnsemblePredictor(
        hpo_results_path="./hpo_logs/hpo_nll_quantile_checkpointing_masked_random_noise/results.csv",
        model_dir="/pscratch/sd/y/yixuans/saved_models_hpo_masked_random_noise/",
        criterion="topK",
        val_data=valset,
        test_data=testSet,
    )

    def process_rollout(data, if_var=False):
        process_data = []
        for d in data:
            d = np.asarray([p.cpu().detach().numpy() for p in d])
            d = predictor.map_to_physical_scale(d, if_var)
            process_data.append(d)
        return np.asarray(process_data)

    SAVE_DIR = "/pscratch/sd/y/yixuans/hpo_uq_pred_random_noise/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # just rollout
    test_rollout_loader = DataLoader(testSet, batch_size=1, shuffle=False)
    model_ids = predictor.select_topK(10)
    rollouts, al_uc, ep_uc = predictor.just_rollout(
        test_rollout_loader, model_ids=model_ids, steps=500
    )
    print(len(rollouts), len(al_uc), len(ep_uc))
    rollouts_remap = process_rollout(rollouts)
    al_uc_remap = process_rollout(al_uc, if_var=True)
    ep_uc_remap = process_rollout(ep_uc, if_var=True)

    # convert to numpy
    rollouts = [r.cpu().numpy() for r in rollouts]
    al_uc = [a.cpu().numpy() for a in al_uc]
    ep_uc = [e.cpu().numpy() for e in ep_uc]

    np.savez(
        f"{SAVE_DIR}/topK_just_rollout.npz",
        rollouts=rollouts,
        al_uc=al_uc,
        ep_uc=ep_uc,
        rollouts_remap=rollouts_remap,
        al_uc_remap=al_uc_remap,
        ep_uc_remap=ep_uc_remap,
    )

    """
    # make predictions with top k
    model_ids = predictor.select_topK(10)
    ensemble_predictions, true_labels = predictor.predict_test(model_ids=model_ids)
    pred_mean, aleatoric_uc, epistemic_uc = predictor.forecast_and_unceartainties(
        ensemble_predictions, true_labels
    )
    true_labels = predictor.map_to_physical_scale(true_labels.cpu().numpy())
    pred_mean = predictor.map_to_physical_scale(pred_mean.cpu().numpy())
    aleatoric_uc = predictor.map_to_physical_scale(
        aleatoric_uc.cpu().numpy(), if_var=True
    )
    epistemic_uc = predictor.map_to_physical_scale(
        epistemic_uc.cpu().numpy(), if_var=True
    )


    # save the predictions
    np.savez(
        f"{SAVE_DIR}/topK_test_ensemble_predictions.npz",
        true_labels=true_labels,
        pred_mean=pred_mean,
        aleatoric_uc=aleatoric_uc,
        epistemic_uc=epistemic_uc,
    )

    # generate rollout
    test_rollout_loader = DataLoader(testSet, batch_size=1, shuffle=False)
    rollouts = predictor.ensemble_rollout(test_rollout_loader, model_ids=model_ids)

   
    true = process_rollout(rollouts["true"])
    pred = process_rollout(rollouts["pred"])
    al_uc = process_rollout(rollouts["al_uc"], if_var=True)
    ep_uc = process_rollout(rollouts["ep_uc"], if_var=True)

    np.savez(
        f"{SAVE_DIR}/topK_test_rollout.npz",
        true=true,
        pred=pred,
        al_uc=al_uc,
        ep_uc=ep_uc,
    )


    # reset using noisy test set for test scores
    testSet = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "test",
        y_noise=True,
        transform=True,
    )
    predictor = EnsemblePredictor(
        hpo_results_path="./hpo_logs/hpo_nll_quantile_checkpointing_masked_SEED/results.csv",
        model_dir="/pscratch/sd/y/yixuans/saved_models_hpo_masked/",
        criterion="topK",
        val_data=valset,
        test_data=testSet,
    )

    # val predict vs num of models
    model_ids = predictor.select_topK(10)
    scores = []
    for i in range(len(model_ids)):
        cur_models = list(model_ids[: i + 1])
        print(cur_models)
        weights = [1 / len(cur_models)] * len(cur_models)
        ensemble_predictions, true_labels = predictor.predict_val(
            model_ids=cur_models, weights=weights
        )
        pred_mean, aleatoric_uc, epistemic_uc = predictor.forecast_and_unceartainties(
            ensemble_predictions, true_labels, weights=weights
        )

        mask = (true_labels != 0).cpu().numpy()
        pred_std = np.sqrt(aleatoric_uc.cpu().numpy())
        score = log_likelihood_score(
            loc=pred_mean.cpu().numpy()[mask],
            scale=pred_std[mask],
            sample=true_labels.cpu().numpy()[mask],
        ) - MSE(pred_mean.cpu().numpy()[mask], true_labels.cpu().numpy()[mask])
        print(score)
        scores.append(score)


    scores = np.asarray(scores)
    np.save(
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/topK_val_num_members_scores.npy",
        scores,
    )

    os.environ["SEED"] = "1"
    # test predict vs num of models
    scores = []
    for i in range(len(model_ids)):
        cur_models = list(model_ids[: i + 1])
        print(cur_models)
        weights = [1 / len(cur_models)] * len(cur_models)
        ensemble_predictions, true_labels = predictor.predict_test(
            model_ids=cur_models, weights=weights
        )
        pred_mean, aleatoric_uc, epistemic_uc = predictor.forecast_and_unceartainties(
            ensemble_predictions, true_labels, weights=weights
        )

        mask = (true_labels != 0).cpu().numpy()
        pred_std = np.sqrt(aleatoric_uc.cpu().numpy())
        score = log_likelihood_score(
            loc=pred_mean.cpu().numpy()[mask],
            scale=pred_std[mask],
            sample=true_labels.cpu().numpy()[mask],
        ) - MSE(pred_mean.cpu().numpy()[mask], true_labels.cpu().numpy()[mask])
        print(score)
        scores.append(score)

    scores = np.asarray(scores)
    np.save(
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/topK_test_num_members_scores.npy",
        scores,
    )

    # use the same test set for greedy test scores
    os.environ["SEED"] = "1"
    data = np.load("caruana_ids_masked.npz")
    model_ids = data["model_ids"]
    # weights = data["weights"]

    scores = []
    for i in range(len(model_ids)):
        cur_models = list(model_ids[: i + 1])
        print(cur_models)
        weights = [1 / len(cur_models)] * len(cur_models)
        ensemble_predictions, true_labels = predictor.predict_test(
            model_ids=cur_models, weights=weights
        )
        pred_mean, aleatoric_uc, epistemic_uc = predictor.forecast_and_unceartainties(
            ensemble_predictions, true_labels, weights=weights
        )

        mask = (true_labels != 0).cpu().numpy()
        pred_std = np.sqrt(aleatoric_uc.cpu().numpy())
        score = log_likelihood_score(
            loc=pred_mean.cpu().numpy()[mask],
            scale=pred_std[mask],
            sample=true_labels.cpu().numpy()[mask],
        ) - MSE(pred_mean.cpu().numpy()[mask], true_labels.cpu().numpy()[mask])
        print(score)
        scores.append(score)

    scores = np.asarray(scores)
    np.save(
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/greedy_test_num_members_scores.npy",
        scores,
    )

    # greedy val
    os.environ["SEED"] = "0"
    scores = []
    for i in range(len(model_ids)):
        cur_models = list(model_ids[: i + 1])
        print(cur_models)
        weights = [1 / len(cur_models)] * len(cur_models)
        ensemble_predictions, true_labels = predictor.predict_val(
            model_ids=cur_models, weights=weights
        )
        pred_mean, aleatoric_uc, epistemic_uc = predictor.forecast_and_unceartainties(
            ensemble_predictions, true_labels, weights=weights
        )

        mask = (true_labels != 0).cpu().numpy()
        pred_std = np.sqrt(aleatoric_uc.cpu().numpy())
        score = log_likelihood_score(
            loc=pred_mean.cpu().numpy()[mask],
            scale=pred_std[mask],
            sample=true_labels.cpu().numpy()[mask],
        ) - MSE(pred_mean.cpu().numpy()[mask], true_labels.cpu().numpy()[mask])
        print(score)
        scores.append(score)


    scores = np.asarray(scores)
    np.save(
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/greedy_val_num_members_scores.npy",
        scores,
    )
    """


if __name__ == "__main__":
    main()

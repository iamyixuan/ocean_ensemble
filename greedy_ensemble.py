import numpy as np
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
        hpo_results_path="./hpo_logs/hpo_nll_quantile_checkpointing_masked_SEED/results.csv",
        model_dir="/pscratch/sd/y/yixuans/saved_models_hpo_masked/",
        criterion="caruana",
        val_data=valset,
        test_data=testSet,
    )
    caruana_ids, caruana_weights = predictor.select_caruana(
        init_K=1, K=10, pred_path="/pscratch/sd/y/yixuans/search_pred_val_masked/"
    )
    print(caruana_ids, caruana_weights)

    np.savez(
        "caruana_ids_masked.npz",
        model_ids=caruana_ids,
        weights=caruana_weights,
    )

    """    
    # load model ids and weights
    data = np.load("caruana_ids_masked.npz")
    model_ids = data["model_ids"]
    weights = data["weights"]

    ensemble_predictions, true_labels = predictor.predict_test(
        model_ids=model_ids, weights=weights
    )

    pred_mean, aleatoric_uc, epistemic_uc = predictor.forecast_and_unceartainties(
        ensemble_predictions, true_labels, weights=weights
    )
    true_labels = predictor.map_to_physical_scale(true_labels.cpu().numpy())
    pred_mean = predictor.map_to_physical_scale(pred_mean.cpu().numpy())
    aleatoric_uc = predictor.map_to_physical_scale(
        aleatoric_uc.cpu().numpy(), if_var=True
    )
    epistemic_uc = predictor.map_to_physical_scale(
        epistemic_uc.cpu().numpy(), if_var=True
    )

    np.savez(
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/greedy_test_ensemble_predictions.npz",
        true_labels=true_labels,
        pred_mean=pred_mean,
        aleatoric_uc=aleatoric_uc,
        epistemic_uc=epistemic_uc,
    )

    test_rollout_loader = DataLoader(testSet, batch_size=1, shuffle=False)
    rollouts = predictor.ensemble_rollout(
        test_rollout_loader, model_ids=model_ids, weights=weights
    )

    def process_rollout(data, if_var=False):
        process_data = []
        for d in data:
            d = np.asarray([p.cpu().detach().numpy() for p in d]).squeeze()
            d = predictor.map_to_physical_scale(d, if_var)
            process_data.append(d)
        return np.asarray(process_data)

    true = process_rollout(rollouts["true"])
    pred = process_rollout(rollouts["pred"])
    al_uc = process_rollout(rollouts["al_uc"], if_var=True)
    ep_uc = process_rollout(rollouts["ep_uc"], if_var=True)

    np.savez(
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/greedy_test_rollout.npz",
        true=true,
        pred=pred,
        al_uc=al_uc,
        ep_uc=ep_uc,
    )
    """

    """
    # reset using noisy test set for test scores
    testSet = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "test",
        y_noise=True,
        transform=True,
    )
    predictor = EnsemblePredictor(
        hpo_results_path="./hpo_logs/hpo_nll_quantile_checkpointing_masked/results.csv",
        model_dir="/pscratch/sd/y/yixuans/saved_models_hpo_masked/",
        criterion="topK",
        val_data=valset,
        test_data=testSet,
    )
    # load model ids and weights
    data = np.load("caruana_ids_masked.npz")
    model_ids = data["model_ids"]
    weights = data["weights"]

    # val predict vs num of models
    # scores = []
    # for i in range(len(model_ids)):
    #     cur_models = list(model_ids[: i + 1])
    #     print(cur_models)
    #     weights = [1 / len(cur_models)] * len(cur_models)

    #     pred_mean, pred_scale, true = predictor._aggregate(
    #         cur_models,
    #         weights,
    #         pred_path="/pscratch/sd/y/yixuans/search_pred_val_masked/",
    #     )

    #     mask = (true != 0).cpu().numpy()

    #     logLikelihood = log_likelihood_score(
    #         loc=pred_mean[mask],
    #         scale=pred_scale[mask],
    #         sample=true.cpu().numpy()[mask],
    #     )
    #     mse_score = MSE(pred_mean[mask], true.cpu().numpy()[mask])
    #     score = logLikelihood - mse_score
    #     print(logLikelihood, mse_score, score)
    #     scores.append(score)

    # scores = np.asarray(scores)
    # np.save(
    #     "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/greedy_val_num_members_scores.npy",
    #     scores,
    # )

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
        "/pscratch/sd/y/yixuans/hpo_uq_pred_masked/greedy_test_num_members_scores.npy",
        scores,
    )
    """


if __name__ == "__main__":
    main()

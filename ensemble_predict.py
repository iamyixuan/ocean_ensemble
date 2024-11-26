import os
import numpy as np
import pandas as pd
import torch
from data import SOMAdata
from metrics import MSE, ll_score, log_likelihood_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_model_nll import FNO_MU_STD as FNO_nll
from train_model_quantile import FNO_QR as FNO_quantile

"""
This script is to form the ensemble and make predicitons on the testing set
There are few functions needed:
1. load the search results and return the list of ensembles based on either
top K or Caruana's method.
2. load the members and make predictions on the testing set,
save all the predictions
and calcuate the ensemble predictions.
3. with the ensemble predictions, calculate the accuracy and other metrics.
4. enable rollout predictions as well.
5. plot the uncertanties and metrics and rollout metrics.
"""


class EnsemblePredictor:
    def __init__(self, hpo_results_path, model_dir, criterion, val_data, test_data):
        self.hpo_results = self.clean_df(pd.read_csv(hpo_results_path))
        self.val_loader = DataLoader(
            val_data, batch_size=val_data.__len__(), shuffle=False
        )
        self.test_loader = DataLoader(
            test_data, batch_size=test_data.__len__(), shuffle=False
        )
        self.model_dir = model_dir
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_y_preds(self, save_path=None):
        model_ids = self.hpo_results["job_id"].values
        models = self.load_models(model_ids, self.model_dir)
        for model, id in tqdm(zip(models, model_ids)):
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                for data in self.val_loader:
                    inputs, y = data
                    inputs = inputs.to(self.device)
                    y = y.to(self.device)
                    outputs = model(inputs)
                    ch = y.shape[1]
                    if outputs.shape[1] == 2 * ch:
                        loc = outputs[:, :ch]
                        scale = torch.sqrt(outputs[:, ch:])
                    elif outputs.shape[1] == 3 * ch:
                        loc = outputs[:, ch : 2 * ch]
                        scale = 0.5 * torch.abs(outputs[:, 2 * ch :] - outputs[:, :ch])
                    preds = {"true": y, "loc": loc, "scale": scale}
                    torch.save(preds, save_path + f"{id}.pt")
        return

    def clean_df(self, df):
        df = df[df["objective_0"] != "F"]
        df.loc[:, "objective_0"] = df["objective_0"].values.astype(float)
        df.loc[:, "objective_1"] = df["objective_1"].values.astype(float)
        df = df.dropna()
        df = df.rename(columns=lambda x: x.replace("p:", ""))
        return df

    def extract_configs(self, job_id):
        df = self.hpo_results.copy()
        index = df[df["job_id"] == job_id].index[0]
        return df.loc[index]

    def select_topK(self, K):
        df = self.hpo_results.copy()
        df["objective_sum"] = df["objective_0"] + df["objective_1"]
        sorted_df = df.sort_values(by="objective_sum", ascending=False)
        return sorted_df["job_id"][:K].values

    def _aggregate(self, model_ids, weights, pred_path):
        pred_loc = []
        pred_scale = []
        for id in model_ids:
            preds = torch.load(f"{pred_path}/{id}.pt")
            pred_loc.append(preds["loc"])
            pred_scale.append(preds["scale"])
        pred_loc = torch.stack(pred_loc).cpu().numpy()
        pred_scale = torch.stack(pred_scale).cpu().numpy()
        weighted_loc = np.average(pred_loc, axis=0, weights=weights)
        weighted_scale = np.average(pred_scale, axis=0, weights=weights)
        return weighted_loc, weighted_scale, preds["true"]

    def select_caruana(self, init_K, K, pred_path):
        """
        potential issues is if the init k = 1 and it's the
        best performing model, the search will stop.
        Therefore, let's set init_K greater than 1
        """
        # start with the best model
        df = self.hpo_results.copy()
        df["objective_sum"] = df["objective_0"] + df["objective_1"]
        sorted_df = df.sort_values(by="objective_sum", ascending=False)
        selected_ids = sorted_df.iloc[:init_K]["job_id"].values.tolist()
        selected_weights = [1 / init_K] * init_K
        print("start with:", selected_ids, "weights: ", selected_weights)

        # the total number is big so select a subset
        # but should it be the top or random?
        available_ids = sorted_df["job_id"].values.tolist()

        cur_loc, cur_scale, true = self._aggregate(
            selected_ids, selected_weights, pred_path
        )

        while len(selected_ids) < K:
            print(f"Selected {len(selected_ids)}/{K} models")
            scores = []  # higher the better
            best_score = -np.inf
            for i in tqdm(available_ids):
                ids = selected_ids + [i]
                weights = [1 / len(ids)] * len(ids)
                weighted_loc, weighted_scale, true = self._aggregate(
                    ids, weights, pred_path
                )
                mask = (true != 0).cpu().numpy()
                score = log_likelihood_score(
                    loc=weighted_loc[mask],
                    scale=weighted_scale[mask],
                    sample=true.cpu().numpy()[mask],
                ) - MSE(weighted_loc[mask], true.cpu().numpy()[mask])
                scores.append(score)

            best_index = np.argmax(scores)

            # if we want to force to fill up the ensemble
            # instead of adding a model that only improves
            # we can sort the performance and add the best
            if scores[best_index] > best_score:
                best_score = scores[best_index]
                print("current best score:", best_score)
            selected_ids.append(available_ids[best_index])
            print("Adding model:", available_ids[best_index])

            # selected_ids, selected_weights = np.unique(
            #     selected_ids, return_counts=True
            # )
            # selected_ids = selected_ids.tolist()
            # selected_weights = selected_weights / np.sum(selected_weights)
            print("Selected ids:", selected_ids)
            # print("Selected weights:", selected_weights)

        weights = [1 / len(selected_ids)] * len(selected_ids)
        return selected_ids, weights

    def ensemble_predict(self, models, data_loader):
        ensemble_predictions = []
        assert len(models) > 0

        for model in models:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                for data in data_loader:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(inputs)
                    ensemble_predictions.append(outputs)

        ensemble_predictions = torch.stack(ensemble_predictions)
        return ensemble_predictions, labels

    def _ensemble_predict(self, models, x, y):
        ensemble_predictions = []
        for model in models:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                outputs = model(x)
                ensemble_predictions.append(outputs)
        ensemble_predictions = torch.stack(ensemble_predictions)
        return ensemble_predictions, y

    def predict_val(self, model_ids, weights=None):
        # """
        # Only use for Caruana's method
        # """
        # assert self.criterion == "caruana"
        models = self.load_models(model_ids, self.model_dir)

        ensemble_predictions, true_labels = self.ensemble_predict(
            models, self.val_loader
        )
        return ensemble_predictions, true_labels

    def predict_test(self, model_ids=None, weights=None):
        # if model_ids is None and self.criterion == "topK":
        #     model_ids = self.select_topK(10)  # select top 10
        # elif self.criterion == "caruana" and model_ids is None:
        #     raise ValueError("Please provide model ids for Caruana's method")

        models = self.load_models(model_ids, self.model_dir)
        ensemble_predictions, true_labels = self.ensemble_predict(
            models, self.test_loader
        )
        return ensemble_predictions, true_labels

    def load_models(self, model_ids, model_dir):
        model_list = []
        for model_id in model_ids:
            config = self.extract_configs(model_id)
            if config["loss"] == "nll":
                model = FNO_nll(
                    in_channels=5,
                    out_channels=4 * 2,
                    decoder_layers=config["num_projs"],
                    decoder_layer_size=config["proj_size"],
                    decoder_activation_fn=config["proj_act"],
                    dimension=2,
                    latent_channels=config["latent_ch"],
                    num_fno_layers=config["num_FNO"],
                    num_fno_modes=int(config["num_modes"]),
                    padding=config["padding"],
                    padding_type=config["padding_type"],
                    activation_fn=config["lift_act"],
                    coord_features=config["coord_feat"],
                )
            elif config["loss"] == "quantile":
                model = FNO_quantile(
                    in_channels=5,
                    out_channels=4 * 3,
                    decoder_layers=config["num_projs"],
                    decoder_layer_size=config["proj_size"],
                    decoder_activation_fn=config["proj_act"],
                    dimension=2,
                    latent_channels=config["latent_ch"],
                    num_fno_layers=config["num_FNO"],
                    num_fno_modes=int(config["num_modes"]),
                    padding=config["padding"],
                    padding_type=config["padding_type"],
                    activation_fn=config["lift_act"],
                    coord_features=config["coord_feat"],
                )

            model.load_state_dict(torch.load(f"{model_dir}/0.{model_id}.pth"))
            model_list.append(model)
        return model_list

    def forecast_and_unceartainties(
        self, ensemble_predictions, true_labels, weights=None
    ):
        assert len(ensemble_predictions.shape) == 5
        if weights is None:
            weights = [1 / ensemble_predictions.shape[0]] * ensemble_predictions.shape[
                0
            ]

        if not isinstance(weights, list):
            weights = weights.tolist()
        assert len(weights) == ensemble_predictions.shape[0], (
            len(weights),
            ensemble_predictions.shape[0],
        )

        ch = true_labels.shape[1]
        ensemble_mean = np.average(
            ensemble_predictions.cpu().numpy(), axis=0, weights=weights
        )
        ensemble_mean = torch.from_numpy(ensemble_mean).float().to(self.device)
        if ensemble_predictions.shape[2] == 2 * ch:
            pred_mean = ensemble_mean[:, :ch]
            pred_var = ensemble_mean[:, ch:]
            aleatoric_uc = pred_var
            epistemic_uc = torch.var(ensemble_predictions, dim=0)[:, :ch]
        elif ensemble_predictions.shape[2] == 3 * ch:
            pred_mean = ensemble_mean[:, ch : 2 * ch]
            pred_std = 0.5 * torch.abs(
                ensemble_mean[:, 2 * ch :] - ensemble_mean[:, :ch]
            )
            aleatoric_uc = pred_std**2
            epistemic_uc = torch.var(ensemble_predictions, dim=0)[
                :, ch : 2 * ch
            ]  # variance of the median prediction
        else:
            raise ValueError("The ensemble predictions are not correct")
        return pred_mean, aleatoric_uc, epistemic_uc

    def map_to_physical_scale(self, x, if_var=False):
        """
        x: (B, 4, 100, 100)
        """
        assert len(x.shape) == 4
        datamin = np.array([34.01481, 5.144762, 3.82e-8, 6.95e-9])
        datamax = np.array([34.24358, 18.84177, 0.906503, 1.640676])
        datamin = datamin[None, :, None, None]
        datamax = datamax[None, :, None, None]
        if if_var:
            x = x * (datamax - datamin) ** 2
        else:
            x = x * (datamax - datamin) + datamin
        return x

    def ensemble_rollout(self, dataloader, model_ids=None, weights=None):
        models = self.load_models(model_ids, self.model_dir)
        weights = [1 / len(model_ids)] * len(model_ids)

        # for every 29 steps we save a sequence
        truePred = {"true": [], "pred": [], "al_uc": [], "ep_uc": []}
        temp_pred = []
        temp_al = []
        temp_ep = []
        true = []
        for i, (x, y) in tqdm(enumerate(dataloader)):
            mask = y[0, 0, :, :] == 0
            mask = np.repeat(mask[None, None, :, :], x.shape[1], axis=1)

            assert x.shape[0] == 1
            x = x.to(self.device)
            x[mask] = 0  # mask the padding
            y = y.to(self.device)
            if i % 29 == 0 and i > 0:
                truePred["true"].append(true)
                truePred["pred"].append(temp_pred)
                truePred["al_uc"].append(temp_al)
                truePred["ep_uc"].append(temp_ep)

                ensemble_predictions, true_labels = self._ensemble_predict(models, x, y)
                pred, al_uc, ep_uc = self.forecast_and_unceartainties(
                    ensemble_predictions, true_labels, weights=weights
                )
                temp_pred = [pred]  # reset sequence
                temp_al = [al_uc]
                temp_ep = [ep_uc]
                true = [y]
            else:
                if len(temp_pred) != 0:
                    x_cat = torch.cat([temp_pred[-1], x[:, -1:]], dim=1)
                    x_cat[mask] = 0  # mask the padding
                    ensemble_predictions, true_labels = self._ensemble_predict(
                        models, x_cat, y
                    )
                    pred, al_uc, ep_uc = self.forecast_and_unceartainties(
                        ensemble_predictions, true_labels, weights=weights
                    )
                    temp_pred.append(pred)
                    temp_al.append(al_uc)
                    temp_ep.append(ep_uc)
                    true.append(y)
                else:
                    ensemble_predictions, true_labels = self._ensemble_predict(
                        models, x, y
                    )
                    pred, al_uc, ep_uc = self.forecast_and_unceartainties(
                        ensemble_predictions, true_labels, weights=weights
                    )
                    temp_pred = [pred]
                    temp_al = [al_uc]
                    temp_ep = [ep_uc]
                    true = [y]
        return truePred

    def just_rollout(self, dataloader, model_ids, steps=100):
        models = self.load_models(model_ids, self.model_dir)
        weights = [1 / len(model_ids)] * len(model_ids)

        x0 = dataloader.dataset[0][0].unsqueeze(0).to(self.device)
        y0 = dataloader.dataset[0][1].unsqueeze(0)
        print(x0.shape, y0.shape)
        mask = y0[0, 0, :, :] == 0
        mask = np.repeat(mask[None, None, :, :], x0.shape[1], axis=1)

        rollouts = []
        al_uc = []
        ep_uc = []
        for t in tqdm(range(steps)):
            if t == 0:
                ensemble_predictions, true_labels = self._ensemble_predict(
                    models, x0, y0
                )
                pred, aleatoric_uc, epistemic_uc = self.forecast_and_unceartainties(
                    ensemble_predictions, true_labels, weights=weights
                )
                # pred[mask] = 0
                rollouts.append(pred)
                al_uc.append(aleatoric_uc)
                ep_uc.append(epistemic_uc)
            else:
                x = rollouts[-1]
                x = torch.cat([x, x0[:, -1:]], dim=1)  # append the parameter
                x[mask] = 0
                ensemble_predictions, true_labels = self._ensemble_predict(
                    models, x, y0
                )
                pred, aleatoric_uc, epistemic_uc = self.forecast_and_unceartainties(
                    ensemble_predictions, true_labels, weights=weights
                )
                rollouts.append(pred)
                al_uc.append(aleatoric_uc)
                ep_uc.append(epistemic_uc)

        return rollouts, al_uc, ep_uc


if __name__ == "__main__":

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
        criterion="topK",
        val_data=valset,
        test_data=testSet,
    )

    # create predictions on the validaiton set
    os.environ["SEED"] = "0"
    predictor.get_y_preds(save_path="/pscratch/sd/y/yixuans/search_pred_val_masked/")

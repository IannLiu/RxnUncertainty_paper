import chemprop
import numpy as np
import os
from typing import List


def predict(smiles_list: List[List[str]],
            ensemble_dir: str,
            num_models: int = 5):
    """
    Return the prediction results of an ensemble model of a fold
    """
    pred_fold = None
    for model_i in range(num_models):
        cp_dir = os.path.join(ensemble_dir, f'model_{model_i}')
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', cp_dir,
            '--uncertainty_method', 'mve',
            '--num_workers', 0,
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args, return_uncertainty=True, smiles=smiles_list)
        if pred_fold is None:
            pred_fold = preds
        else:
            pred_fold = np.concatenate([pred_fold, np.array(preds)], axis=-1)

    # squeeze to 2D for save
    value = pred_fold[0, :, :]
    unc = pred_fold[1, :, :]
    pred_value = np.mean(value, axis=-1, keepdims=True)
    epi = np.var(value, axis=-1, keepdims=True)
    ale = np.mean(unc, axis=-1, keepdims=True)
    tot = ale + epi
    data = np.concatenate([pred_value, epi, ale, tot], axis=-1)

    return data


if __name__ == '__main__':
    rxns = [
        ['[C:1]([C:2](=[O:3])[H:7])([H:4])([H:5])[H:6].[C:8]([O:9][C:10]([C:11]([H:15])([H:16])[H:17])=[O:12])([H:13])[H:14]>>[C:1]([C:2]=[O:3])([H:4])([H:5])[H:6].[H:7][C:8]([O:9][C:10]([C:11]([H:15])([H:16])[H:17])=[O:12])([H:13])[H:14]'],
    ]

    prediction = predict(smiles_list=rxns, ensemble_dir='../unc_models/default/ccsdt/ensemble/fold_0')
    print(prediction)

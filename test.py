import numpy as np
from sklearn.metrics import r2_score


def test(model, test_loader, device):
    pred = []
    real = []
    for batch_idx, data in enumerate(test_loader):
        data.to(device)
        out = model(data)

        pred.extend(out.detach().numpy().tolist())
        real.extend(data.y.detach().numpy().tolist())

    return np.mean((np.array(real) - np.array(pred)) ** 2), \
        r2_score(y_pred=pred, y_true=real)

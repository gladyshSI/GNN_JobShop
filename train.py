import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from test import test


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def train(data_list, model, optimizer, loss_function, device, num_folds=1, num_epochs_per_fold=200):
    NUM_FOLDS = num_folds
    NUM_EPOCHS_PER_FOLD = num_epochs_per_fold

    if NUM_FOLDS == 1:
        n = len(data_list)
        train_size = int(np.round(n * 0.8))
        splits = [(list(range(train_size)), list(range(train_size, n)))]
    else:
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
        splits = kfold.split(data_list)

    # pbar = tqdm(total=NUM_FOLDS * NUM_EPOCHS_PER_FOLD)

    for kfold_idx, (train_index, test_index) in enumerate(splits):
        train_loader = DataLoader(
            [data_list[d] for d in train_index], batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            [data_list[d] for d in test_index], batch_size=32, shuffle=True
        )

        reset_weights(model)
        model.train()

        epochs = []
        test_r2 = []
        train_r2 = []
        test_loss_mean2 = []
        train_loss_mean2 = []
        for epoch in range(NUM_EPOCHS_PER_FOLD):  # tqdm(range(NUM_EPOCHS_PER_FOLD)):
            for batch_idx, data in enumerate(train_loader):
                data.to(device)
                optimizer.zero_grad()

                output = model(data)
                # print("output.shape() =", output.size())
                y = data.y
                # print("y.shape() =", y.size())
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                sq_mean, r2 = test(model, test_loader, device)
                sq_mean_train, r2_train = test(model, train_loader, device)
                epochs.append(epoch)
                test_loss_mean2.append(sq_mean)
                train_loss_mean2.append(sq_mean_train)
                test_r2.append(r2)
                train_r2.append(r2_train)
                print("epoch =", epoch, "sq_mean =", sq_mean, "r2 =", r2)

            if epoch % 100 == 0:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

                ax[0].plot(epochs, test_r2, label='test r2')
                ax[0].plot(epochs, train_r2, label='train r2')
                ax[0].legend(loc='best')
                ax[1].plot(epochs, test_loss_mean2, label='test mean2')
                ax[1].plot(epochs, train_loss_mean2, label='train mean2')
                ax[1].legend(loc='best')
                plt.show()
        sq_mean, r2 = test(model, test_loader, device)
        print("sq_mean =", sq_mean, "r2 =", r2)

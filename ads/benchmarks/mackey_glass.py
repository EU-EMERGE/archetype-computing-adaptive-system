import torch


def get_mackey_glass(csvpath, lag=84, washout=200):
    with open(csvpath, 'r') as f:
        dataset = f.readlines()[0]

    # 10k steps
    dataset = torch.tensor([float(el) for el in dataset.split(',')]).float()

    end_train = int(dataset.shape[0] / 2)
    end_val = end_train + int(dataset.shape[0] / 4)
    end_test = dataset.shape[0]

    train_dataset = dataset[:end_train-lag]
    train_target = dataset[washout+lag:end_train]

    val_dataset = dataset[end_train:end_val-lag]
    val_target = dataset[end_train+washout+lag:end_val]

    test_dataset = dataset[end_val:end_test-lag]
    test_target = dataset[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)
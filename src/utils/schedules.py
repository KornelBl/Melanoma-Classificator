def LR_test_schedule(epoch, lr):
    lr = 1e-10 * 3**epoch
    return lr

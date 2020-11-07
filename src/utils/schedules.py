


def CLR_schedule(epoch,lr):
    lr = 1e-10 * 3**epoch
    return lr
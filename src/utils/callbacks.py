import tensorflow.keras as keras


class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_auc',
             min_delta=0, patience=0, verbose=0, mode='max', start_epoch = 100): # add argument for starting epoch
        super(CustomStopper, self).__init__(monitor=monitor, mode=mode, patience=patience)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
        else:
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
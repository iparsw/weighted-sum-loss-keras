from tensorflow import keras

class WS_loss(keras.losses.Loss):
    def __init__(self, loss_fns: list = None,
                 loss_fns_weight: list = None,
                 name='weighted_sum'):
        super().__init__(name=name)
        self.loss_fns = loss_fns
        self.loss_fns_weight = loss_fns_weight

    def call(self, y_true, y_pred):
        loss_value = 0
        counter = 0
        for loss_function in self.loss_fns:
            loss_value += loss_function(y_true, y_pred) * self.loss_fns_weight[counter]
            counter += 1
        return loss_value

    def get_config(self):
        return {"loss_fns": self.loss_fns,
                "loss_fns_weight": self.loss_fns_weight}
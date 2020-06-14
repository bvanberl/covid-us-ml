from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.utils import get_custom_objects


class DifferentialAdam(Adam):
    """Optimizer wrapper for per layer learning rate.
    This wrapper is used to add per layer learning rates by
    providing per layer factors which are multiplied with the
    learning rate of the optimizer.
    Note: This is a wrapper and does not implement any
    optimization algorithm.
    # Arguments
        optimizer: An optimizer class to be wrapped.
        lr_multipliers: Dictionary of the per layer factors. For
            example `optimizer={'conv_1/kernel':0.5, 'conv_1/bias':0.1}`.
            If for kernel and bias the same learning rate is used, the
            user can specify `optimizer={'conv_1':0.5}`.
        **kwargs: The arguments for instantiating the wrapped optimizer
            class.
    """
    def __init__(self, optimizer, lr_multipliers=None, **kwargs):
        super(DifferentialAdam, self).__init__(name='DifferentialAdam', **kwargs)
        self._lr_multipliers = lr_multipliers or {}

    def _get_multiplier(self, param):
        for k in self._lr_multipliers.keys():
            if k in param.name:
                return self._lr_multipliers[k]

    def get_updates(self, loss, params):
        mult_lr_params = {p: self._get_multiplier(p) for p in params
                          if self._get_multiplier(p)}
        base_lr_params = [p for p in params if self._get_multiplier(p) is None]

        updates = []
        base_lr = self.learning_rate
        for param, multiplier in mult_lr_params.items():
            self.learning_rate = base_lr * multiplier
            updates.extend(self.get_updates(loss, [param]))

        self.learning_rate = base_lr
        updates.extend(self.get_updates(loss, base_lr_params))

        return updates

    def get_config(self):
        config = {'lr_multipliers': self._lr_multipliers}
        base_config = super(DifferentialAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'DifferentialAdam': DifferentialAdam})
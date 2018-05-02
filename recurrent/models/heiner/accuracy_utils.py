from keras.layers import Layer
import keras.backend as K
from heiner.utils import mask_from
from keras import metrics


class StatefulMetric(Layer):

    def __init__(self, metric, output_threshold, mask_val, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.stateful = True
        self.metric_value = K.variable(value=0, dtype='int32')

        self.metric = metric
        self.output_threshold = output_threshold
        self.mask_val = mask_val

    def reset_states(self):
        K.set_value(self.metric_value, 0)

    def __call__(self, y_true, y_pred):
        y_pred_labels = K.cast(K.greater_equal(y_pred, self.output_threshold), 'float32')
        mask, count_unmasked = mask_from(y_true, self.mask_val)

        if self.metric == 'TP':
            true_positives = K.sum(y_pred_labels * y_true * mask)
            metric_value = true_positives
        elif self.metric == 'TN':
            true_negatives = K.sum((y_pred_labels - 1) * (y_true - 1) * mask)
            metric_value = true_negatives
        elif self.metric == 'P':
            count_positives = K.sum(y_true * mask)  # just the +1 labels are added, the rest is 0
            count_positives = K.switch(count_positives, count_positives, 1.0)
            metric_value = count_positives
        elif self.metric == 'N':
            count_positives = K.sum(y_true * mask)  # just the +1 labels are added, the rest is 0
            count_positives = K.switch(count_positives, count_positives, 1.0)

            count_negatives = count_unmasked - count_positives  # count_unmasked are all valid labels
            count_negatives = K.switch(count_negatives, count_negatives, 1.0)
            metric_value = count_negatives
        else:
            raise ValueError("metric has to be either 'TP', 'TN', 'P' or 'N'")

        current_metric_value = self.metric_value * 1
        metric_value = K.cast(metric_value, 'int32')
        self.add_update(K.update_add(self.metric_value,
                                     metric_value),
                        inputs=[y_true, y_pred])
        return current_metric_value + metric_value


def stateful_metric_builder(metric, output_threshold, mask_val):
    metric_names = {'TP': 'true_positives', 'TN': 'true_negatives', 'P': 'positives', 'N': 'negatives'}
    if metric not in metric_names.keys():
        raise ValueError("metric has to be either 'TP', 'TN', 'P' or 'N'")

    metric_fn = StatefulMetric(metric, output_threshold, mask_val, metric_names[metric])
    config = metrics.serialize(metric_fn)
    config['config']['metric'] = metric
    config['config']['output_threshold'] = output_threshold
    config['config']['mask_val'] = mask_val
    metric_fn = metrics.deserialize(config, custom_objects={'StatefulMetric': StatefulMetric})
    return metric_fn

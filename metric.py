# from collections import defaultdict
import re

import torch
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_auprc,
    binary_f1_score,  # F1
    binary_precision,  # precision
)


def create_metrics(metric_names: list, **kwargs):
    metrics = [METRICS_NAMES[m] for m in metric_names]

    return combine_metrics(metrics)(**kwargs)


def combine_metrics(metrics: list):
    class CombineMetrics(*metrics):
        def __init__(self, **kwargs):
            for m_obj in metrics:
                m_obj.__init__(self, **kwargs)

    return CombineMetrics


class BaseMetric:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, **kwargs):

        full_metric_result = {}

        for name in dir(self):
            if not name.startswith("get_metric_"):
                continue
            stat_name = re.sub("get_metric_", "", name)
            method = getattr(self, name)
            metric = method(**kwargs)  # should be returning a dict
            full_metric_result.update(metric)

        return full_metric_result

    # def create_metric_string(self, metric_result, flag="Train"):
    #    return flag + " " + " | ".join([f"{k}: {v}" for k, v in metric_result.items()])


class F1Metric(BaseMetric):
    def __init__(self, thresholds=[0.1, 0.5, 0.9], **kwargs):
        self.thresholds = thresholds

    # def get_pred_labels(self, predictions):
    #    return

    def get_metric_f1(self, predictions=None, labels=None, **kwargs):
        result = {
            f"f1": float(
                multiclass_f1_score(predictions, labels, num_classes=2)
                .detach()
                .cpu()
                .numpy()
            )
        }
        return result


class PrecisionMetric(BaseMetric):
    def __init__(self, thresholds=[0.1, 0.5, 0.9], **kwargs):
        self.thresholds = thresholds

    def get_metric_precision(self, predictions=None, labels=None, **kwargs):
        result = {
            "precision": float(
                multiclass_auprc(predictions, labels).detach().cpu().numpy()
            )
        }

        return result


METRICS_NAMES = {"F1": F1Metric, "precision": PrecisionMetric}

import torch
import torch.nn.functional as F

class Metrics():
    def __init__(self, **kwargs):
        """Class containing all metrics
        """
        self.metrics = {k: v for k, v in kwargs.items()}

    def apply_threshold(self, y, threshold=0.5):
        return (y > threshold).to(torch.uint8)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, loss, pred, y):
        pass

    def compute(self):
        return {k: v.compute().item() for k, v in self.metrics.items()}

    def to(self, device):
        self.metrics = {k: v.to(device) for k, v in self.metrics.items()}


class MetricsDenoise(Metrics):
    def __init__(self, **kwargs):
        """
        Class for computing metrics for denoising tasks.

        Args:
            **kwargs: Additional keyword arguments representing the metrics to be computed.

        Attributes:
            metrics (dict): A dictionary containing the metrics to be computed.

        Methods:
            apply_threshold: Applies a threshold to the predicted values.
            update: Updates the metrics with the given loss, predicted values, and ground truth values.
            reset: Resets the metrics to their initial state.
            compute: Computes the metrics and returns the results.
            to: Moves the metrics to the specified device.
        """
        super().__init__(**kwargs)

    def update(self, loss_total, pred, y):
        if type(pred) is tuple: # if a tuple of predictions from the D(z) - original latent space and D(z') - denoised latent space
            pred_latent, pred_original = pred
            pred_latent_th = self.apply_threshold(pred_latent)
            pred_th = self.apply_threshold(pred_original)
        else:
            pred_th = self.apply_threshold(pred)
            
        if type(loss_total) is tuple:
            loss = loss_total[0]
        else:
            loss = loss_total

        y_th = self.apply_threshold(y)

        for metric_name in self.metrics.keys():
            if metric_name == "loss":
                self.metrics["loss"].update(loss)
            elif metric_name == "miou":
                self.metrics["miou"].update(pred_th, y_th)
            elif metric_name == "miou_latent":
                self.metrics["miou_latent"].update(pred_latent_th, y_th)
            
def print_metrics(epoch, train_metrics, val_metrics, time):
    """Print all metrics

    Args:
        epoch (int): epoch numbre
        train_metrics (list of Metrics): list train of metrics
        val_metrics (list of Metrics): list validation of metrics
        time (float): time (s)
    """
    str_ = f'>> epoch: {epoch} ({time:.2f}s)'
    for (split, metric) in zip(['train', 'valid'], [train_metrics, val_metrics]):
        if metric is None:
            continue
        str_ += "\n"
        str_ += f'[{split}]'
        for k, v in metric.items():
            if k == 'loss':
                str_ += f' | {k}: {v.item():.8f}'
            else:
                str_ += f' | {k}: {v.item():.4f}'
    print(str_)


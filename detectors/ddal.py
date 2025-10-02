from .base import SemisupervisedDriftDetector, BatchDetector

import sys
from metrics.computational_metrics import computational_metrics


class DDAL(SemisupervisedDriftDetector, BatchDetector):
    """
    A Drift Detection Method Based on Active Learning.
    
    DDAL is a concept drift detection method based on  
    density variation of the most significant instances selected for Active Learning.
    
    More information:
        https://ieeexplore.ieee.org/document/8489364
        
        Albert França Josuá Costa
        Regis Antonio Saraiva Albuquerque
        Eulanda Miranda dos Santos
        
    Parameters
    ----------
    batch_size
        Size of instances batch.
    
    theta
        Drift threshold.

    lambida
        Uncertainty threshold.
        
    Methods
    ----------
    fixed_uncertainty
        Compute the uncertainty of the current instance.
    
    count_selected_instances
        Check if current instances fall in the virtual margin.
    
    compute_current_density
        Compute the current density into the current batch.
    
    detection_modulo
        Check if drift ocurred.
        
    reset
        Reset the detector.
    
    """

    def __init__(self, batch_size: int = 10, theta: float = 0.00005,
                 lambida: float = 0.75, seed=None,
                 recent_samples_size: int = 500):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         batch_size=batch_size)
        self.theta = theta
        self.lambida = lambida
        self.max_density = sys.float_info.min
        self.min_density = sys.float_info.max
        self.current_density = 0.0
        self.amount_selected_instances = 0
        self.batch_size = batch_size

    def fixed_uncertainty(self, maximum_posteriori):
        selected = False
        if maximum_posteriori < self.lambida:
            selected = True
        return selected

    def count_selected_instances(self, maximum_posteriori):
        s = self.fixed_uncertainty(maximum_posteriori)
        if s:
            self.amount_selected_instances += 1

    def count_tensor(self, tensor):
        for val in tensor:
            s = self.fixed_uncertainty(val)
            if s:
                self.amount_selected_instances += 1

    def compute_current_density(self):
        self.current_density = (float)(
            self.amount_selected_instances / self.batch_size)

    def detection_module(self):

        isDrift = False

        if self.current_density > self.max_density:
            self.max_density = self.current_density

        if self.current_density < self.min_density:
            self.min_density = self.current_density

        if (self.max_density - self.min_density) > self.theta:
            isDrift = True

        return isDrift

    def reset(self, batch_size: int = 500, theta: float = 0.95,
              lambida: float = 0.95):
        self.theta = theta
        self.lambida = lambida
        self.max_density = sys.float_info.min
        self.min_density = sys.float_info.max
        self.current_density = 0.0
        self.amount_selected_instances = 0
        self.batch_size = batch_size

    def update(self, x) -> bool:
        """
        Updates the detector with a new instance and checks for drift.
        
        Parameters
        ----------
        x : array-like
            The input features for the instance.
        
        y_true : int or None
            The true label of the instance (optional).
        
        Returns
        -------
        bool
            True if drift is detected, False otherwise.
        """

        # Compute current density after processing the batch
        self.compute_current_density()

        # Check for drift detection
        if self.detection_module():
            self.reset()
            return True

    @computational_metrics
    def process_main_batch_stream(self, stream, n_training_samples: int,
                                  classifier):
        for batch_id, batch in enumerate(self.batch_stream(
                stream, n_training_samples)):
            for i, (x, y) in enumerate(batch):
                self.predictions.append(classifier.predict(x))
                self.labels.append(y)

                y_pred = classifier.predict_proba(x)
                max_y_pred_prob = max(y_pred.values())
                self.count_selected_instances(max_y_pred_prob)

            if self.update(batch):  # Use the drift detector's update method
                # 
                self.handle_batch_update(batch_id, n_training_samples)
                for i, (x, y) in enumerate(batch):
                    classifier.fit(x, y)

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_batch_stream(stream, n_training_samples,
                                        classifier_path)

def get_portion_requested_labels(n_samples, n_req_labels,
                                 n_training_samples) -> float:
    """
    Count the labels used while processing the stream.
    """
    return n_req_labels / (n_samples - n_training_samples)

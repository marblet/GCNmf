import random
import numpy as np
import torch
import datetime
import logging
import random
from sklearn import metrics


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(f"Random seed set as {seed}")


"""Project wide logger."""


# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create the Handler for logging data to a file
now = datetime.datetime.now()
prefix_time = now.strftime("%Y%m%d_%H%M%S")
filename = f'log/{prefix_time}_{random.randint(0, 99999)}.log'
logger_handler = logging.FileHandler(filename=filename)
logger_handler.setLevel(logging.DEBUG)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
logger.addHandler(logger_handler)

# Add stream handler in same format
logger_handler = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)
logger_handler.setLevel(logging.DEBUG)
logger.addHandler(logger_handler)

# Make globally available
LOGGER_FILENAME = filename

logger.info(
  f'Completed configuring logger; writing from {__name__} to {filename}')


def bootstrap_sampling_ave_precision(predictions, outcome, num_samples=12):
  """Bootstrap samples the performance metrics."""
  results_roc = np.zeros((num_samples))
  results_ap = np.zeros((num_samples))

  for i in range(num_samples):
    if num_samples == 1:
      ind = np.arange(predictions.shape[0])
    else:
      ind = np.random.choice(
        predictions.shape[0], predictions.shape[0], replace=True)

    results_roc[i] = metrics.roc_auc_score(
      y_true=outcome[ind], y_score=predictions[ind])
    results_ap[i] = metrics.average_precision_score(
      y_true=outcome[ind], y_score=predictions[ind])

  q20, q50_auroc, q80 = np.quantile(results_roc, [0.2, 0.5, 0.8])*100
  logger.info(f"ROC: {q50_auroc:5.1f} [{q20:5.1f}, {q80:5.1f}]")

  q20, q50_ap, q80 = np.quantile(results_ap, [0.2, 0.5, 0.8])*100
  logger.info(f"AvP: {q50_ap:5.1f} [{q20:5.1f}, {q80:5.1f}]")
  return q50_auroc, q50_ap
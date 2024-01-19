"""Utility functions for WandB."""
from utils import logger
import os
import psutil
import subprocess
import time


class WandbDummy:
  """Dummy class for WandB."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    self.name = "dummy"

  def log(self, *args, **kwargs):
    """Dummy logging function."""
    del args, kwargs

  def finish(self, *args, **kwargs):
    """Dummy finish function."""
    del args, kwargs

  def get_url(self, *args, **kwargs):
    """Dummy get_url function."""
    del args, kwargs
    return "https://wandb.ai/martentyrk/GCNmf/"


def make_git_log():
  """Logs the git diff and git show.

  Note that this function has a general try/except clause and will except most
  errors produced by the git commands.
  """
  try:
    result = subprocess.run(
      ['git', 'status'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git status \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(
      ['git', 'show', '--summary'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git show \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git diff \n{result.stdout.decode('utf-8')}")
  except Exception as e:  # pylint: disable=broad-except
    logger.info(f"Git log not printed due to {e}")


def log_to_wandb(wandb_runner):
  """Logs system statistics to wandb every minute."""
  while True:
    loadavg1, loadavg5, _ = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)

    wandb_runner.log({
      "loadavg1": loadavg1,
      "loadavg5": loadavg5,
      "swap_use": swap_use})
    time.sleep(60)

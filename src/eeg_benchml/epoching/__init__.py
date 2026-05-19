"""Fixed-length epoching, automatic rejection, and training-only augmentation."""

from .augmentation import AugmentationConfig, augment_training_epochs
from .rejection import RejectionConfig, reject_bad_epochs
from .segment import EpochingConfig, make_fixed_length_epochs

__all__ = [
    "EpochingConfig",
    "make_fixed_length_epochs",
    "RejectionConfig",
    "reject_bad_epochs",
    "AugmentationConfig",
    "augment_training_epochs",
]

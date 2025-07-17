from .instance.bear import *
from .instance.turtle import *
from .instance.python import *
from .instance.panther import *
from .instance.gibon import *

DATASET_NAME_MAPPING = {
    "bear": BearHugDataset,
    "turtle": TurtleHugDataset,
    "python": PythonHugDataset,
    "panther": PantherHugDataset,
    "gibon": GibonHugDataset,

}

T2I_DATASET_NAME_MAPPING = {
    "bear": BearHugDatasetForT2I,
    "turtle": TurtleHugDatasetForT2I,
    "python": PythonHugDatasetForT2I,
    "panther": PantherHugDatasetForT2I,
    "gibon": GibonHugDatasetForT2I,
}
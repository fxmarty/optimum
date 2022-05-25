from dataclasses import field
from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Extra

from . import is_pydantic_available

if is_pydantic_available():
    from pydantic.dataclasses import dataclass
else:
    from dataclasses import dataclass

class APIFeaturesManager:
    _SUPPORTED_TASKS = ["text-classification", "token-classification", "question-answering"]

    @staticmethod
    def check_supported_model_task_pair(model_type: str, task: str):
        model_type = model_type.lower()
        if model_type not in APIFeaturesManager._SUPPORTED_MODEL_TYPE:
            raise KeyError(
                f"{model_type} is not supported yet. "
                f"Only {list(APIFeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        elif task not in APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]:
            raise KeyError(
                f"{task} is not supported yet for model {model_type}. "
                f"Only {APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]} are supported. "
                f"If you want to support {task} please propose a PR or open up an issue."
            )

    @staticmethod
    def check_supported_task(task: str):
        if task not in APIFeaturesManager._SUPPORTED_TASKS:
            raise KeyError(
                f"{task} is not supported yet. "
                f"Only {APIFeaturesManager._SUPPORTED_TASKS} are supported. "
                f"If you want to support {task} please propose a PR or open up an issue."
            )

class BaseModelNoExtra(BaseModel):
    class Config:
        extra = Extra.forbid  # ban additional arguments

class Frameworks(str, Enum):
    onnxruntime = "onnxruntime"


class CalibrationMethods(str, Enum):
    minmax = "minmax"
    percentile = "percentile"
    entropy = "entropy"


class QuantizationApproach(str, Enum):
    static = "static"
    dynamic = "dynamic"


@dataclass
class Calibration:
    """Parameters for post-training calibration with static quantization."""

    method: CalibrationMethods = field(metadata={"help": "hehehehe"})
    num_calibration_samples: int = field()
    calibration_histogram_percentile: Optional[float] = field(
        default=None,
    )
    calibration_moving_average: Optional[bool] = field(
        default=None,
    )
    calibration_moving_average_constant: Optional[float] = field(
        default=None,
    )


@dataclass
class FrameworkArgs:
    opset: Optional[int] = field(default=15)
    optimization_level: Optional[int] = field(default=0)


@dataclass
class Versions:
    transformers: str = field()
    optimum: str = field()
    optimum_hash: Optional[str]
    onnxruntime: Optional[str]
    torch_ort: Optional[str]


@dataclass
class Evaluation:
    time: List[Dict]
    others: Dict


@dataclass
class DatasetArgs:
    """Parameters related to the dataset."""

    path: str = field()
    eval_split: str = field()
    data_keys: Dict[str, Union[None, str]] = field()
    ref_keys: List[str] = field()
    name: Optional[str] = field(default=None)
    calibration_split: Optional[str] = field(default=None)


@dataclass
class TaskArgs:
    """Task-specific parameters."""

    is_regression: Optional[bool] = field(default=None)


@dataclass
class RunConfig():
    """Parameters defining a run. A run is an evaluation of a triplet (model, dataset, metric) coupled with optimization parameters, allowing to compare a transformers baseline and a model optimized with Optimum."""

    model_name_or_path: str = field()
    task: str = field()
    quantization_approach: QuantizationApproach = field()
    dataset: DatasetArgs = field()
    framework: Frameworks = field()
    framework_args: FrameworkArgs = field()
    batch_sizes: Optional[List[int]] = field(default_factory=lambda: [4, 8])
    input_lengths: Optional[List[int]] = field(default_factory=lambda: [128])
    operators_to_quantize: Optional[List[str]] = field(default_factory=lambda: ["Add", "MatMul"])
    node_exclusion: Optional[List[str]] = field(default_factory=lambda: [])
    
    metrics: List[str] = field(default="accuracy")
    per_channel: Optional[bool] = field(default=False)
    calibration: Optional[Calibration] = field(default=None)
    task_args: Optional[TaskArgs] = field(default=None)
    aware_training: Optional[bool] = field(default=False)


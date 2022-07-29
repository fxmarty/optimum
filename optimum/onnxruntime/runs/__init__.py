import copy
import json
import os
import shutil
from typing import Dict, Optional, Union

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType

from ...pipelines import pipeline as _optimum_pipeline
from ...runs_base import Run, TimeBenchmark, get_autoclass_name, task_processing_map
from .. import ORTQuantizer
from ..configuration import ORTConfig, QuantizationConfig
from ..modeling_ort import ORTModel
from ..preprocessors import QuantizationPreprocessor
from .calibrator import OnnxRuntimeCalibrator
from .utils import task_ortmodel_map


class OnnxRuntimeRun(Run):
    def __init__(self, run_config):
        run_config = super().__init__(run_config)

        # Create the quantization configuration containing all the quantization parameters
        qconfig = QuantizationConfig(
            is_static=self.static_quantization,
            format=QuantFormat.QDQ if self.static_quantization else QuantFormat.QOperator,
            mode=QuantizationMode.QLinearOps if self.static_quantization else QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QInt8 if self.static_quantization else QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=run_config["per_channel"],
            reduce_range=False,
            operators_to_quantize=run_config["operators_to_quantize"],
        )

        quantizer = ORTQuantizer.from_pretrained(
            run_config["model_name_or_path"],
            feature=get_autoclass_name(self.task),
            opset=run_config["framework_args"]["opset"],
        )

        self.ort_config = ORTConfig(opset=quantizer.opset, quantization=qconfig)

        self.preprocessor = copy.deepcopy(quantizer.preprocessor)

        self.batch_sizes = run_config["batch_sizes"]
        self.input_lengths = run_config["input_lengths"]

        self.time_benchmark_args = run_config["time_benchmark_args"]

        self.model_path = os.path.join(self.run_dir_path, "model.onnx")
        self.quantized_model_path = os.path.join(self.run_dir_path, "quantized_model.onnx")

        processing_class = task_processing_map[self.task]
        self.task_processor = processing_class(
            dataset_path=run_config["dataset"]["path"],
            dataset_name=run_config["dataset"]["name"],
            calibration_split=run_config["dataset"]["calibration_split"],
            eval_split=run_config["dataset"]["eval_split"],
            preprocessor=self.preprocessor,
            data_keys=run_config["dataset"]["data_keys"],
            ref_keys=run_config["dataset"]["ref_keys"],
            task_args=run_config["task_args"],
            static_quantization=self.static_quantization,
            num_calibration_samples=run_config["calibration"]["num_calibration_samples"]
            if self.static_quantization
            else None,
            config=quantizer.model.config,
            max_eval_samples=run_config["max_eval_samples"],
        )

        self.metric_names = run_config["metrics"]

        self.load_datasets()

        quantization_preprocessor = QuantizationPreprocessor()
        ranges = None
        if self.static_quantization:
            calibration_dataset = self.get_calibration_dataset()
            calibrator = OnnxRuntimeCalibrator(
                calibration_dataset,
                quantizer,
                self.model_path,
                qconfig,
                calibration_params=run_config["calibration"],
                node_exclusion=run_config["node_exclusion"],
            )
            ranges, quantization_preprocessor = calibrator.calibrate()

        # Export the quantized model
        quantizer.export(
            onnx_model_path=self.model_path,
            onnx_quantized_model_output_path=self.quantized_model_path,
            calibration_tensors_range=ranges,
            quantization_config=qconfig,
            preprocessor=quantization_preprocessor,
        )

        # onnxruntime benchmark
        ort_session = ORTModel.load_model(self.quantized_model_path)

        # necessary to pass the config for the pipeline not to complain later
        self.ort_model = task_ortmodel_map[self.task](ort_session, config=quantizer.model.config)

        self.return_body[
            "model_type"
        ] = quantizer.model.config.model_type  # return_body is initialized in parent class

    def _launch_time(self, trial):
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        input_length = trial.suggest_categorical("input_length", self.input_lengths)

        model_input_names = set(self.preprocessor.model_input_names)

        # onnxruntime benchmark
        print("Running ONNX Runtime time benchmark.")
        time_benchmark = TimeBenchmark(
            self.ort_model,
            input_length=input_length,
            batch_size=batch_size,
            model_input_names=model_input_names,
            warmup_runs=self.time_benchmark_args["warmup_runs"],
            duration=self.time_benchmark_args["duration"],
        )
        time_metrics = time_benchmark.execute()

        time_evaluation = {
            "batch_size": batch_size,
            "input_length": input_length,
        }
        time_evaluation.update(time_metrics)

        self.return_body["evaluation"]["time"].append(time_evaluation)

        return 0, 0

    def launch_eval(
        self, save: bool = False, save_directory: Union[str, os.PathLike] = None, run_name: Optional[str] = None
    ):
        kwargs = self.task_processor.get_pipeline_kwargs()

        # transformers pipelines are smart enought to detect whether the tokenizer or feature_extractor is needed
        ort_pipeline = _optimum_pipeline(
            task=self.task,
            model=self.ort_model,
            tokenizer=self.preprocessor,
            feature_extractor=self.preprocessor,
            accelerator="ort",
            **kwargs,
        )

        eval_dataset = self.get_eval_dataset()

        print("Running evaluation...")
        metrics_dict = self.task_processor.run_evaluation(eval_dataset, ort_pipeline, self.metric_names)

        metrics_dict.pop("total_time_in_seconds", None)
        metrics_dict.pop("samples_per_second", None)
        metrics_dict.pop("latency_in_seconds", None)

        self.return_body["evaluation"]["others"].update(metrics_dict)

        if save:
            self.save(save_directory, run_name)

        return self.return_body

    def save(self, save_directory: Union[str, os.PathLike], run_name: str):
        save_directory = super().save(save_directory, run_name)

        # save ORTConfig
        self.ort_config.save_pretrained(save_directory)

        # save non-quantized and quantized models
        shutil.move(self.model_path, os.path.join(save_directory, "model.onnx"))
        shutil.move(self.quantized_model_path, os.path.join(save_directory, "quantized_model.onnx"))

        # save run config and evaluation results
        with open(os.path.join(save_directory, "results.json"), "w") as f:
            json.dump(self.return_body, f, indent=4)

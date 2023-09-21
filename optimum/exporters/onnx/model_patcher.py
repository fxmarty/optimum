# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

from ...utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

    from .base import OnnxConfig

logger = logging.get_logger(__name__)


def get_argument(argument_name: str, args: List[Any], kwargs: Dict[str, Any], forward_signature):
    """
    Get the argument argument_name from the args and kwargs according to the signature forward_signature.
    """
    args = list(args)
    if argument_name in forward_signature.parameters:
        argument_index = list(forward_signature.parameters.keys()).index(argument_name)
        return args[argument_index]
    else:
        return kwargs[argument_name]


def override_arguments(args, kwargs, forward_signature, model_kwargs: Dict[str, Any]):
    """
    Override the args and kwargs with the argument values from model_kwargs, following the signature forward_signature corresponding to args and kwargs.
    """
    args = list(args)

    for argument in model_kwargs:
        if argument in forward_signature.parameters:
            argument_index = list(forward_signature.parameters.keys()).index(argument)
            if argument in kwargs or len(args) <= argument_index:
                kwargs[argument] = model_kwargs[argument]
            else:
                args[argument_index] = model_kwargs[argument]
        else:
            kwargs[argument] = model_kwargs[argument]

    return args, kwargs


@dataclasses.dataclass
class PatchingSpec:
    """
    Data class that holds patching specifications.

    Args:
        o: Module / object where the op to patch is located
        name: Name of the op to monkey patch
        custom_op: Custom op that patches the original op
        orig_op: Original op that is being patched
        op_wrapper: Wrapper (optional) that wraps both the original and custom ops.
            It is useful for ops that are class or static methods for instance.
    """

    o: Any
    name: str
    custom_op: Callable
    orig_op: Optional[Callable] = None
    op_wrapper: Optional[Callable] = None


class ModelPatcher:
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._model = model

        patching_specs = config.PATCHING_SPECS
        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # TODO: remove that once we got rid of OnnxConfigWithLoss or we implemented it better.
        if config.__class__.__name__ == "OnnxConfigWithLoss":
            self.real_config = config._onnx_config
        else:
            self.real_config = config

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            outputs = self.orig_forward(*args, **kwargs)

            # This code block handles different cases of the filterd_outputs input to align it with the expected
            # format of outputs. It is common for the output type of a model to vary, such as tensor, list,
            # tuple, etc. For Transformers models, the output is encapsulated in a ModelOutput object that
            # contains the output names of the model. In the case of Timm classification models, the output
            # is of type tensor. By default, it is assumed that the output names mentioned in the ONNX config
            # match the outputs in order.
            filterd_outputs = {}
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                    if (
                        onnx_output_name in config.outputs
                        or (allow_past_in_outputs and name.startswith("past_key_values"))
                        or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                    ):
                        filterd_outputs[name] = value
            elif isinstance(outputs, (list, tuple)):
                outputs_list = list(config.outputs.keys())
                dict(zip(outputs_list, outputs))
            else:
                if len(config.outputs) > 1:
                    num_outputs = len(config.outputs)
                    outputs_str = ", ".join(config.outputs.keys())
                    raise ValueError(
                        f"config.outputs should have only one outputs, but it has {num_outputs} keys: {outputs_str}"
                    )
                else:
                    name = list(config.outputs.keys())[0]
                    filterd_outputs[name] = outputs
                name = list(config.outputs.keys())[0]
                filterd_outputs[name] = outputs
            return filterd_outputs

        self.patched_forward = patched_forward

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    def __enter__(self):
        self.patch_ops()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)

    def __call__(self, *args, **kwargs):
        if getattr(self._model, self.orig_forward_name) is self.orig_forward:
            logger.warning("Running the non-patched model")
        return self._model(*args, **kwargs)


class Seq2SeqModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        # use_cache is by default set to False with pix2struct, so we need to set it to
        # True to export with past key value
        if model.config.model_type == "pix2struct" and allow_past_in_outputs:
            model.config.text_config.use_cache = True

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            outputs = self.orig_forward(*args, **kwargs)

            # Filter out cross attention past key values output from the decoder using KV cache, as they are constants.
            filterd_outputs = {}
            for name, value in outputs.items():
                onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                if (
                    onnx_output_name in config.outputs
                    or (allow_past_in_outputs and name.startswith("past_key_values"))
                    or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                ):
                    if name != "past_key_values":
                        if self.real_config._behavior == "decoder" and name == "encoder_last_hidden_state":
                            # Who cares about the encoder outputs in the decoder?
                            continue
                        else:
                            filterd_outputs[name] = value
                    else:
                        if self.real_config._behavior == "monolith" or (
                            self.real_config._behavior == "decoder"
                            and (self.real_config.is_merged or not self.real_config.use_past_in_inputs)
                        ):
                            filterd_outputs[name] = value
                        elif self.real_config._behavior == "decoder" and self.real_config.use_past_in_inputs:
                            # The filtering happens here. The decoder with use_past_in_inputs=True corresponds to the autoregressive one.
                            filterd_outputs[name] = tuple([v[:2] for v in value])

            return filterd_outputs

        self.patched_forward = patched_forward


class WavLMModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            model_kwargs = self.model_kwargs
            # setting output_attentions=True in the model input to avoid calling torch.nn.functional.scaled_dot_product_attention
            # in https://github.com/huggingface/transformers/blob/v4.27.1/src/transformers/models/wavlm/modeling_wavlm.py#L496
            # that calls https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/functional.py#L5334
            model_kwargs["output_attentions"] = True
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=model_kwargs)

            outputs = self.orig_forward(*args, **kwargs)

            filterd_outputs = {}
            for name, value in outputs.items():
                onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                if (
                    onnx_output_name in config.outputs
                    or (allow_past_in_outputs and name.startswith("past_key_values"))
                    or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                ):
                    filterd_outputs[name] = value
            return filterd_outputs

        self.patched_forward = patched_forward


class SAMModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        def patched_forward(
            pixel_values=None,
            input_points=None,
            image_embeddings=None,
            image_positional_embeddings=None,
            return_dict=True,
            **kwargs,
        ):
            if config.variant == "monolith":
                return self.orig_forward(
                    pixel_values=pixel_values,
                    input_points=input_points,
                    image_embeddings=image_embeddings,
                    return_dict=return_dict,
                    **kwargs,
                )
            elif config.variant == "split":
                if config.vision_encoder:
                    image_positional_embeddings = model.get_image_wide_positional_embeddings()

                    # repeat with batch size
                    batch_size = pixel_values.shape[0]
                    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

                    vision_outputs = model.vision_encoder(
                        pixel_values,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=return_dict,
                    )
                    image_embeddings = vision_outputs[0]

                    if not return_dict:
                        return (image_embeddings, image_positional_embeddings)
                    else:
                        return {
                            "image_embeddings": image_embeddings,
                            "image_positional_embeddings": image_positional_embeddings,
                        }
                else:
                    if input_points is not None:
                        input_labels = torch.ones_like(
                            input_points[:, :, :, 0], dtype=torch.int, device=input_points.device
                        )
                    else:
                        raise ValueError("input_points is required to export the prompt encoder / mask decoder.")

                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        input_points=input_points,
                        input_labels=input_labels,
                        input_boxes=None,  # Not supported in the ONNX export
                        input_masks=None,  # Not supported in the ONNX export
                    )

                    low_res_masks, iou_predictions, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_positional_embeddings=image_positional_embeddings,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,  # Not supported in the ONNX export
                        attention_similarity=None,  # Not supported in the ONNX export
                        target_embedding=None,  # Not supported in the ONNX export
                        output_attentions=False,
                    )

                    if not return_dict:
                        return (iou_predictions, low_res_masks)
                    else:
                        return {"iou_scores": iou_predictions, "pred_masks": low_res_masks}

        self.patched_forward = patched_forward


class SpeechT5ModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        def patched_forward(
            input_ids=None,
            speaker_embeddings=None,
            encoder_outputs=None,
            past_key_values=None,
            output_sequence=None,
            spectrogram=None,
        ):
            use_cache = self.real_config.use_past and self.real_config.variant == "with-past"
            if self.real_config._behavior == "encoder":
                encoder_attention_mask = torch.ones_like(input_ids)

                encoder_out = model.speecht5.encoder(
                    input_values=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True,
                )
                # downsample encoder attention mask
                if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
                    encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
                        encoder_out[0].shape[1], encoder_attention_mask
                    )

                # TODO: that is wrong?
                return {"encoder_out": encoder_out, "encoder_attention_mask": encoder_attention_mask}

            elif self.real_config._behavior == "decoder" and self.real_config.use_past_in_inputs:
                encoder_hidden_states = encoder_outputs.last_hidden_state

                decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

                # Run the decoder layers on the last element of the prenet output.
                decoder_out = model.speecht5.decoder.wrapped_decoder(
                    hidden_states=decoder_hidden_states[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=False,
                    return_dict=True,
                )

                last_decoder_output = decoder_out.last_hidden_state[0, -1]
                past_key_values = decoder_out.past_key_values

                # Predict the new mel spectrum for this step in the sequence.
                spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
                spectrum = spectrum.view(model.config.reduction_factor, model.config.num_mel_bins)

                # NOTE: extending the spectrogram should is to be handled outside of the ONNX.
                # spectrogram.append(spectrum)

                # Extend the output sequence with the new mel spectrum.
                output_sequence = torch.cat(
                    (output_sequence, spectrum[-1].view(1, 1, model.config.num_mel_bins)), dim=1
                )

                # Predict the probability that this is the stop token.
                prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

                return {
                    "prob": prob,
                    "output_sequence": output_sequence,
                    "spectrum": spectrum
                    # TODO: PKV here
                }
            elif self.real_config.is_postnet_and_vocoder:
                # NOTE: the following concatenation is expected to be handled outside of the ONNX:
                # spectrogram = torch.cat(spectrogram, dim=0).unsqueeze(0)
                spectrogram = spectrogram.unsqueeze(0)
                spectrogram = model.speech_decoder_postnet.postnet(spectrogram)
                spectrogram = spectrogram.squeeze(0)

                waveform = model_kwargs["vocoder"](spectrogram)

                return {"waveform": waveform}

        self.patched_forward = patched_forward

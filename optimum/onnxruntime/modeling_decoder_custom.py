import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import onnxruntime
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from ..exporters import TasksManager
from ..exporters.onnx import export_models, get_decoder_models_for_export
from ..onnx.utils import _get_external_data_paths
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.file_utils import validate_file_exists
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .io_binding import TypeHelper
from .modeling_ort import ORTModel
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, get_provider_for_device, parse_device
from .modeling_decoder import ORTModelDecoder

if TYPE_CHECKING:
    from transformers import PretrainedConfig


if check_if_transformers_greater("4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin


logger = logging.getLogger(__name__)

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

CAUSALLM_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"

TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    >>> text = "My name is Arthur and I live in"
    >>> gen = onnx_gen(text)
    ```
"""

DECODER_ONNX_FILE_PATTERN = r"(.*)?decoder((?!with_past).)*?\.onnx"
DECODER_WITH_PAST_ONNX_FILE_PATTERN = r"(.*)?decoder(.*)?with_past(.*)?\.onnx"


class CustomORTDecoder:
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        config: "PretrainedConfig",
        device: torch.device,
        use_io_binding: Optional[bool] = None,
    ):
        self.session = session
        self.config = config
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(self.config.model_type)(
            self.config
        )
        self._device = device
        self.use_io_binding = use_io_binding
        self.session_inputs = {output_key.name: idx for idx, output_key in enumerate(self.session.get_inputs())}
        self.session_outputs = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.session_input_names = list(self.session_inputs.keys())
        self.session_output_names = list(self.session_outputs.keys())
        # TODO: make this less hacky.
        self.key_value_input_names = [key for key in self.session_input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [
            key for key in self.session_output_names if (".key" in key) or (".value" in key)
        ]
        self.name_to_np_type = TypeHelper.get_io_numpy_type_map(self.session) if self.use_io_binding else None
        self.num_pkv = 2  # (self-attention key and value per decoder layer)

    def generate_past_example(self, batch_size, method='zeros'):
        methods = {
            'randn': torch.randn,
            'ones': torch.ones,
            'zeros': torch.zeros
            # 'softmax':
        }
        emb_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        shape = [batch_size, self.config.num_attention_heads, 1, emb_per_head]
        past_inputs = [[methods[method](shape) for _ in range(self.num_pkv)] for _ in range(self.normalized_config.num_layers)]
        return past_inputs

    def prepare_output_buffer(
        self,
        output_name,
        batch_size=None,
        sequence_length=None,
        past_sequence_length=None,
    ):
        """
        Prepare the buffer of outputs(`logits`/`key_values`/`loss`) with 1D tensors.
        """
        ort_type = TypeHelper.get_output_type(self.session, output_name)
        torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
        if output_name == "logits":
            output_shape = (batch_size, sequence_length, self.normalized_config.vocab_size)
            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()
        elif ".key" in output_name or ".value" in output_name:
            num_attention_heads = self.normalized_config.num_attention_heads
            hidden_size = self.normalized_config.hidden_size
            embed_size_per_head = hidden_size // num_attention_heads

            if past_sequence_length is not None:
                sequence_length += past_sequence_length
            output_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)

            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()

        return output_shape, output_buffer

    def prepare_io_binding(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ):
        io_binding = self.session.io_binding()

        # Bind the inputs

        # Bind input ids
        input_ids = input_ids.contiguous()
        io_binding.bind_input(
            "input_ids",
            input_ids.device.type,
            self._device.index,
            self.name_to_np_type["input_ids"],
            tuple(input_ids.shape),
            input_ids.data_ptr(),
        )

        # Bind the attention mask
        attention_mask = attention_mask.contiguous()
        io_binding.bind_input(
            "attention_mask",
            attention_mask.device.type,
            self._device.index,
            self.name_to_np_type["attention_mask"],
            tuple(attention_mask.shape),
            attention_mask.data_ptr(),
        )

        # Bind the past key values
        if past_key_values is not None:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                past_key_value = past_key_value.contiguous()
                io_binding.bind_input(
                    input_name,
                    past_key_value.device.type,
                    self._device.index,
                    self.name_to_np_type[input_name],
                    tuple(past_key_value.shape),
                    past_key_value.data_ptr(),
                )

        # Bind the outputs

        # Bind the logits
        logits_shape, logits_buffer = self.prepare_output_buffer(
            output_name="logits",
            batch_size=input_ids.size(0),
            sequence_length=input_ids.size(1),
        )
        io_binding.bind_output(
            "logits",
            logits_buffer.device.type,
            self._device.index,
            self.name_to_np_type["logits"],
            logits_shape,
            logits_buffer.data_ptr(),
        )
        output_shapes = {"logits": logits_shape}
        output_buffers = {"logits": logits_buffer}

        # Bind the past keys values
        for key_value_output_name in self.key_value_output_names:
            self_pkv_shape, self_pkv_buffer = self.prepare_output_buffer(
                output_name=key_value_output_name,
                batch_size=input_ids.size(0),
                sequence_length=input_ids.size(1),
                past_sequence_length=past_key_values[0].size(2)
                if past_key_values
                else None,  # sequence length of self-attention key for layer.0
            )
            io_binding.bind_output(
                key_value_output_name,
                self_pkv_buffer.device.type,
                self._device.index,
                self.name_to_np_type[key_value_output_name],
                self_pkv_shape,
                self_pkv_buffer.data_ptr(),
            )
            # set -1 for sequence_length as it could be larger than the real sequence_length for creating buffer
            self_pkv_shape = self_pkv_shape[:2] + (-1,) + self_pkv_shape[3:]
            output_shapes[key_value_output_name] = self_pkv_shape
            output_buffers[key_value_output_name] = self_pkv_buffer

        return io_binding, output_shapes, output_buffers

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = [past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer]

        if self._device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids, attention_mask, past_key_values
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer(2)
            past_key_values = tuple()
            for name in self.key_value_output_names:
                past_key_values += (output_buffers[name].view(output_shapes[name]),)

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (self-attention key and value per decoder layer)
            past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = {
                "input_ids": input_ids.cpu().detach().numpy(),
                "attention_mask": attention_mask.cpu().detach().numpy(),
            }

            if past_key_values is not None:
                # Add the past_key_values to the decoder inputs
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 for the self-attention)
            past_key_values = tuple(
                torch.from_numpy(outputs[self.session_outputs[key]]).to(self._device)
                for key in self.key_value_output_names
            )

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # per decoder layer
            past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
            logits = torch.from_numpy(outputs[self.session_outputs["logits"]]).to(self._device)

        return CausalLMOutputWithCrossAttentions(logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)




class CustomORTModelForCausalLM(ORTModelDecoder, GenerationMixin):
    """
    ONNX model with a causal language modeling head for ONNX Runtime inference.
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

    @add_start_docstrings_to_model_forward(
        CAUSALLM_ONNX_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCausalLM",
            checkpoint="optimum/gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        print("----")
        print("input_ids.shape", input_ids.shape)
        print("attention_mask.shape", attention_mask.shape)
        print("past_key_values[0][0].shape", past_key_values[0][0].shape)

        """
        if past_key_values is None or self.decoder_with_past is None:
            outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
        """
        outputs = self.decoder_with_past(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        print("outputs.logits shape", outputs.logits.shape)
        print("outputs.logits", outputs.logits[0][0][100:105])
        print("outputs.logits argmax", outputs.logits[0][0].argmax())
        return CausalLMOutputWithCrossAttentions(logits=outputs.logits, past_key_values=outputs.past_key_values)

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly

        attention_mask = kwargs.get("attention_mask", None)  # input_ids.new_ones(input_ids.shape)
        use_cache = kwargs.get("use_cache", None)

        if not past_key_values:
            past_key_values = self.decoder.generate_past_example(input_ids.size(0))

        attention_mask = torch.ones([input_ids.shape[0], input_ids.shape[1] + 1], dtype=torch.int64)
        attention_mask[:, 0] = 0

        return {
            "input_ids": input_ids if past_key_values[0][0].size(2) == 1 else input_ids[:, -1:],
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

# Copyright 2026 The LiteRT Torch Authors.
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
# ==============================================================================
"""Exportable modules for extended modules."""

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.model_ext.gemma3 import vision_exportable as gemma3_vision_exportable
from litert_torch.generative.export_hf.model_ext.gemma3n import exportable_module as gemma3n_exportable
from litert_torch.generative.export_hf.model_ext.gemma3n import vision_exportable as gemma3n_vision_exportable
import transformers


def get_prefill_decode_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig,
):
  """Gets prefill-decode exportables."""
  if model_config.model_type == 'gemma3n':
    assert (
        not export_config.split_cache
    ), 'Split cache is not supported for Gemma3N.'
    assert (
        export_config.externalize_embedder
    ), 'External embedder is required for Gemma3N.'
    print('Using Gemma3N exportables.')
    return (
        gemma3n_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
        gemma3n_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
    )
  else:
    pass
  return None


def get_vision_exportables(
    model_config: transformers.PretrainedConfig,
):
  """Gets vision exportables."""
  if model_config.model_type == 'gemma3':
    return (
        gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionEncoder,
        gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionAdapter,
    )
  elif model_config.model_type == 'gemma3n':
    return (
        gemma3n_vision_exportable.LiteRTExportableModuleForGemma3nVisionEncoder,
        gemma3n_vision_exportable.LiteRTExportableModuleForGemma3nVisionAdapter,
    )
  else:
    raise ValueError(f'Unsupported model type: {model_config.model_type}')


def get_additional_exportables(
    model_config: transformers.PretrainedConfig,
):
  """Gets additional exportables."""
  if model_config.model_type == 'gemma3n':
    return {
        'per_layer_embedder': (
            gemma3n_exportable.LiteRTExportableModuleForPerLayerEmbedder
        ),
    }
  else:
    pass
  return {}

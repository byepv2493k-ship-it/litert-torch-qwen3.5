"""Convert Qwen 3.5 to tflite."""
from absl import app
from litert_torch.generative.examples.qwen import qwen3_5
from litert_torch.generative.utilities import converter
flags = converter.define_conversion_flags("qwen3_5")
_MODEL_SIZE = flags.DEFINE_enum("model_size", "2b", ["2b"], "Model size.")
_BUILDER = {"2b": qwen3_5.build_2b_model}
def main(_):
  converter.build_and_convert_to_tflite_from_flags(_BUILDER[_MODEL_SIZE.value])
if __name__ == "__main__":
  app.run(main)

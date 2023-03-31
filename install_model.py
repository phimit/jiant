import jiant.proj.main.export_model as export_model
import sys

model_name = sys.argv[1]

export_model.export_model(
    hf_pretrained_model_name_or_path=model_name,
    output_base_path=f"exp/models/{model_name}",
)

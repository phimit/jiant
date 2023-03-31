import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os

JIANT_ROOT_FOLDER="/usr/src/app/jiant"
TASK_CONFIG_PATH=[f"{JIANT_ROOT_FOLDER}/jiant/tasks/lib/diapuncmentation/punctuation_config.json", 
                  f"{JIANT_ROOT_FOLDER}/jiant/tasks/lib/diapuncmentation/daseg_config.json"]
RUN_CONFIG_PATH="/usr/src/run_configs/run_config.json"
MODEL_NAME="bert-base-uncased"
MODEL_ROOT_PATH=f"/usr/src/models/{MODEL_NAME}"
CACHE_FOLDER="./cache"
OUTPUT_PATH="/usr/src/output"

print("Downloading model")
export_model.export_model(
    hf_pretrained_model_name_or_path=MODEL_NAME,
    output_base_path=MODEL_ROOT_PATH,
)

# Tokenize and cache each task

for task_name, config_path in zip(["punctuation", "daseg"], TASK_CONFIG_PATH):
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=config_path,
        hf_pretrained_model_name_or_path=MODEL_NAME,
        output_dir=f"{CACHE_FOLDER}/{task_name}",
        phases=["train", "val"],
    ))

    #row = caching.ChunkedFilesDataCache(f"./cache/{task_name}/train").load_chunk(0)[0]["data_row"]
    #print(row.meta)

print("Creating configuration")

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"{JIANT_ROOT_FOLDER}/jiant/tasks/lib/diapuncmentation/",
    task_cache_base_path=CACHE_FOLDER,
    train_task_name_list=["punctuation", "daseg"],
    train_batch_size=64,
    val_task_name_list=["punctuation", "daseg"],
    eval_batch_size=8,
    epochs=10, # -> 10
    num_gpus=2,
).create_config()
os.makedirs(os.path.dirname(RUN_CONFIG_PATH), exist_ok=True)
py_io.write_json(jiant_run_config, RUN_CONFIG_PATH)
display.show_json(jiant_run_config)


run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path=RUN_CONFIG_PATH,
    output_dir=f"{OUTPUT_PATH}/run1",
    hf_pretrained_model_name_or_path=MODEL_NAME,
    model_path="f{MODEL_ROOT_PATH}/model/model.p",
    model_config_path="f{MODEL_ROOT_PATH}/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=500,
    do_train=True,
    do_val=True,
    force_overwrite=True,
)



main_runscript.run_loop(run_args)
TODO list jiant/expés segmentation

- Disrpt23 priorities
  x- correct split alignment errors: tur, eng.pdtb, rus, zho, eus
  x- saved predictions fail for pdtb ? 
  x- launch model in prediction mode on tests
  x- read hyperparams from config files for stat collections
  /- merge corpus (cf script Mortezza dans discut21/contextual_embeddings/merge.sh and preprocess in main.sh)
        x - do simple pretraining (eg all spa) + fine-tuning (separate evals)
        x- do sequential learning keeping only the encoder between different languages
            eg spa1 -> spa2 -> por1 -> por2 -> fra etc
           cf examples/notebooks/jiant_STILTs_Example.ipynb
  x- test zero shot on ted + merged
        cf examples/notebooks/jiant_XLNI_Example.ipynb
  /- update score collection 
       ?- for merged corpora
       ?- for MT
  x- launch seg_eval for final scores

Experimental tests
  x- test batch size / roberta base (16, 32, 64) / large 32 ?
  x- test freeze more layers roberta large and batch size ?
   - test roberta-large with higher batches / no freezing on g5k / longer max sequence length


  - Adapters, juste use: 
  model = AutoAdapterModel.from_pretrained(args.transformer_model) 
  config = AutoConfig.from_pretrained("xlm-roberta-base")
  active_adapter = model.load_adapter(adapter_name,                                   
                        config = adapter_name + "/adapter_config.json") 
                        or config = config
  model.set_active_adapters(active_adapter)


  - train from checkpoint
    trainer args
       run_with_continue
       save_checkpoint_every_steps=args.save_checkpoint_every_steps -> checkpoint.p
       delete_checkpoint_if_done (no)

  ========================
  Adapters: notebook example for mad-x / adapter huggingface doc/lib/github

  from transformers import AutoConfig, AutoAdapterModel

    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
    )
    model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
    config=config,
    )
    from transformers import AdapterConfig

# Load the language adapters
lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
model.load_adapter("zh/wiki@ukp", config=lang_adapter_config)
model = AutoAdapterModel.from_pretrained(args.transformer_model) 
active_adapter = model.load_adapter(adapter_name,                     
config = adapter_name + "/adapter_config.json") model.set_active_adapters(active_adapter)

model.train_adapter(["copa"])
# Unfreeze and activate stack setup
model.active_adapters = Stack("en", "copa")
===========================



- More complete configurations 
    ?- for each task training (eg max seq length) separate from model definition in task config
        cf script configurator
    x- explicitely stored for each run ? oui mais seulement dernier run dans le dir 
        -> change save config in main script
    x- ou mettre / sampling strategies pour MTL cf jiant/proj/main/components/task_sampler.py
        cf plus bas liste des samplers
    
    
- MTL / controles  
    - test zero-shot multilingue: https://github.com/nyu-mll/jiant/blob/master/examples/notebooks/jiant_XNLI_Example.ipynb
        cf config call en annexe
    - test ajout préfixe de chaque tache à l'input pour test en zero-shot 
    - taches auxilliaires ?    
        - pos-tag ?
        - pred ponctuation ?  
    - freeze backbone + unfreeze backbone

- Baselines
    x- segmentation: no pred / juste début phrase
    - ?conn: projection de lexique

- Evaluation
    - matrice de confusion en meme temps que génération des prédictions 
    - tester alignement prédiction pour connecteurs détection (semble ok vu les resultats)
    - analyses d'erreur quantitatives

- datasets:
    x- disrpt conllu: rien à changer par corpus (phrases trop longues ?)
        plain: 
            - cut à la lecture tous les N tokens, ou bien 
            X- preprocess (ersatz/trankit/spacy) et faire un corpus séparé
        
    - dialogue: 
      x  icsi, 
      x  ami, 
        ?elitr 

- Contrôle des modèles: 
    - comparer autres modèles ? 
        - mBert cased
        - mBART ? implique changer pas mal de trucs
        x- xlm large -
        - try adapaters for better fine-tuning ? https://github.com/adapter-hub/adapter-transformers
          allège aussi l'empreinte mémoire. 
          aussi: LORA 
        - modèle pretrained on downstream data, cf Downstream Datasets Make Surprisingly Good Pretraining Corpora
            https://arxiv.org/abs/2209.14389
    - hyperparam opti
        x- freeze layers (low/high ?)
    - presegmentation morphologique (cf https://aclanthology.org/2022.acl-short.43.pdf) 
                An Embarrassingly Simple Method to  Mitigate und es ira ble  Properties of Pretrained Language Model Tokenizers
    x- ajout rnn: 
        [ok] lstm 
        ? gru (pas nécessaire?)
    - plug modèle préentrainé différent ? 
        - séquence de modèles : cf sequential training example (stilt)
            train a model, keep the backbone but not the head and continue
        - backbone quelconque: oui on peut utiliser un checkpoint d'un encodeur intermédiaire au lieu d'un modèle HF. cf guide/project
            (aussi sequential learning, mais avec résultat d'un training jiant)
    - en amont ajout de features: 
        - cnn de chars ? pas sur très important
        - postag ?

x- Collecte résultats: 
    

Non prioritaire
------------------   
   
Annexes
-------------

- Samplers: 
   
   ProportionalMultiTaskSampler

   SpecifiedProbMultiTaskSampler  : attend un dict tache:poids

   TemperatureMultiTaskSampler :
   			attend
				- dict tache: num instances
				- temperature (float)	
				- optionel: example cap (int)

   TimeDependentProbMultiTaskSampler
   attend
	- dict tache: expr numerique = f(t) (cf numexpr library)
	- max steps

- ex Config call 
configurator: 
jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path="./tasks/configs",
    task_cache_base_path="./cache",
    train_task_name_list=["mnli"],
    val_task_name_list=["mnli", "xnli_de", "xnli_zh"],
    train_batch_size=32,
    eval_batch_size=64,
    epochs=0.1,
    num_gpus=1,
).create_config()



To briefly go over the major components of the jiant_task_container_config:

    task_config_path_dict: The paths to the task config files we wrote above.
    task_cache_config_dict: The paths to the task features caches we generated above.
    sampler_config: Determines how to sample from different tasks during training.
    global_train_config: The number of total steps and warmup steps during training.
    task_specific_configs_dict: Task-specific arguments for each task, such as training batch size and gradient accumulation steps.
    taskmodels_config: Task-model specific arguments for each task-model, including what tasks use which model.
    metric_aggregator_config: Determines how to weight/aggregate the metrics across multiple tasks.

- zero shot: use task model map
    jiant_run_config["taskmodels_config"]["task_to_taskmodel_map"] = {
    "mnli": "nli_model",
    "xnli_de": "nli_model",
    "xnli_zh": "nli_model",
}
os.makedirs("./run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, "./run_configs/jiant_run_config.json")


- STILT exemples: 
fine tuning / STILT / cf x-trem eval
in script calls: 
replace --model_load_mode from_transformers \

with
        --ZZoverrides model_load_path \
        --model_load_mode partial \
        --model_load_path /path/to/my/model.p \

option of what part ? metarunner ?
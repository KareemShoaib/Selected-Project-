---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:47402
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 18x 14gabl vent brand_18
  sentences:
  - hampton bay diner 2light brush nickel fluoresc flush mount -discontinu brand_hampton
  - cellwood 14in. x 14in. white squar gabl vent brand_cellwood color_whit
  - everbilt 14x 34 in. zinc-plat steel pan-head phillip sheet metal screw 8pack brand_everbilt
- source_sentence: swag light kit brand_swag
  sentences:
  - westinghous 18ft. brush nickel swag light kit brand_westinghous
  - salsburi industri 3500seri aluminum recessed-mount privat vertic mailbox with
    7door brand_salsburi
  - complet toilet repair kit brand_complet
- source_sentence: new electr water heater brand_new
  sentences:
  - rheem perform 50gal. tall 6year 45004500watt element electr water heater brand_rheem
  - sea gull light classico 1light polish brass outdoor hang pendant fixtur brand_sea
  - black decker linefind orbit jigsaw with smartselect technolog brand_black+deck
    color_black
- source_sentence: eler outlet plate brand_eler
  sentences:
  - gree high effici 24000 btu 2ton ductless duct free mini split air condition with
    invert heat remot 208230v brand_gre
  - hampton bay step 1duplex outlet plate - age bronz brand_hampton
  - singer confid sew machin brand_sing
- source_sentence: kohler highlin touch biscuit brand_kohl
  sentences:
  - lexan thermoclear 48in. x 96in. x 58 in. bronz multiwal polycarbon sheet brand_lexan
  - vento clover 54in. chrome indoor ceil fan with 3transluc yellow blade brand_vento
    color_yellow
  - kohler highlin comfort height elong toilet bowl in biscuit brand_kohl
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: val
      type: val
    metrics:
    - type: pearson_cosine
      value: 0.5271053959513253
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.5196782645468574
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'kohler highlin touch biscuit brand_kohl',
    'kohler highlin comfort height elong toilet bowl in biscuit brand_kohl',
    'vento clover 54in. chrome indoor ceil fan with 3transluc yellow blade brand_vento color_yellow',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `val`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.5271     |
| **spearman_cosine** | **0.5197** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 47,402 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 10.46 tokens</li><li>max: 27 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 24.74 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.69</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                  | sentence_1                                                                                            | label                            |
  |:--------------------------------------------|:------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code>concret step cap brand_concret</code> | <code>paveston rockwal 3.4in. x 9in. pecan retain concret garden wall block cap brand_paveston</code> | <code>0.665</code>               |
  | <code>hillman group brand_th</code>         | <code>hillman group 14 in. x 212 in. toggl strap with screw 6pack brand_th</code>                     | <code>0.835</code>               |
  | <code>ryobi trimmer brand_ryobi</code>      | <code>ryobi replac trimmer cordless spool cap brand_ryobi</code>                                      | <code>0.33499999999999996</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | val_spearman_cosine |
|:------:|:----:|:-------------:|:-------------------:|
| 0.0337 | 100  | -             | 0.3862              |
| 0.0675 | 200  | -             | 0.4180              |
| 0.1012 | 300  | -             | 0.4276              |
| 0.1350 | 400  | -             | 0.4387              |
| 0.1687 | 500  | 0.0637        | 0.4289              |
| 0.2025 | 600  | -             | 0.4501              |
| 0.2362 | 700  | -             | 0.4569              |
| 0.2700 | 800  | -             | 0.4521              |
| 0.3037 | 900  | -             | 0.4574              |
| 0.3375 | 1000 | 0.0576        | 0.4525              |
| 0.3712 | 1100 | -             | 0.4576              |
| 0.4050 | 1200 | -             | 0.4651              |
| 0.4387 | 1300 | -             | 0.4623              |
| 0.4725 | 1400 | -             | 0.4693              |
| 0.5062 | 1500 | 0.0563        | 0.4611              |
| 0.5400 | 1600 | -             | 0.4732              |
| 0.5737 | 1700 | -             | 0.4712              |
| 0.6075 | 1800 | -             | 0.4797              |
| 0.6412 | 1900 | -             | 0.4809              |
| 0.6750 | 2000 | 0.0567        | 0.4758              |
| 0.7087 | 2100 | -             | 0.4715              |
| 0.7425 | 2200 | -             | 0.4913              |
| 0.7762 | 2300 | -             | 0.4857              |
| 0.8100 | 2400 | -             | 0.4824              |
| 0.8437 | 2500 | 0.0547        | 0.4935              |
| 0.8775 | 2600 | -             | 0.4881              |
| 0.9112 | 2700 | -             | 0.4955              |
| 0.9450 | 2800 | -             | 0.4929              |
| 0.9787 | 2900 | -             | 0.4925              |
| 1.0    | 2963 | -             | 0.4910              |
| 1.0125 | 3000 | 0.0546        | 0.4894              |
| 1.0462 | 3100 | -             | 0.5002              |
| 1.0800 | 3200 | -             | 0.4953              |
| 1.1137 | 3300 | -             | 0.4923              |
| 1.1475 | 3400 | -             | 0.4976              |
| 1.1812 | 3500 | 0.0489        | 0.4919              |
| 1.2150 | 3600 | -             | 0.5038              |
| 1.2487 | 3700 | -             | 0.4983              |
| 1.2825 | 3800 | -             | 0.4970              |
| 1.3162 | 3900 | -             | 0.5071              |
| 1.3500 | 4000 | 0.0497        | 0.5062              |
| 1.3837 | 4100 | -             | 0.5073              |
| 1.4175 | 4200 | -             | 0.5071              |
| 1.4512 | 4300 | -             | 0.5069              |
| 1.4850 | 4400 | -             | 0.4994              |
| 1.5187 | 4500 | 0.0512        | 0.4999              |
| 1.5525 | 4600 | -             | 0.5065              |
| 1.5862 | 4700 | -             | 0.5089              |
| 1.6200 | 4800 | -             | 0.5121              |
| 1.6537 | 4900 | -             | 0.5103              |
| 1.6875 | 5000 | 0.0497        | 0.5079              |
| 1.7212 | 5100 | -             | 0.5072              |
| 1.7550 | 5200 | -             | 0.5093              |
| 1.7887 | 5300 | -             | 0.5106              |
| 1.8225 | 5400 | -             | 0.5100              |
| 1.8562 | 5500 | 0.0496        | 0.5126              |
| 1.8900 | 5600 | -             | 0.5038              |
| 1.9237 | 5700 | -             | 0.5103              |
| 1.9575 | 5800 | -             | 0.5128              |
| 1.9912 | 5900 | -             | 0.5083              |
| 2.0    | 5926 | -             | 0.5094              |
| 2.0250 | 6000 | 0.0495        | 0.5127              |
| 2.0587 | 6100 | -             | 0.5163              |
| 2.0925 | 6200 | -             | 0.5130              |
| 2.1262 | 6300 | -             | 0.5133              |
| 2.1600 | 6400 | -             | 0.5143              |
| 2.1937 | 6500 | 0.0456        | 0.5128              |
| 2.2275 | 6600 | -             | 0.5162              |
| 2.2612 | 6700 | -             | 0.5134              |
| 2.2950 | 6800 | -             | 0.5179              |
| 2.3287 | 6900 | -             | 0.5181              |
| 2.3625 | 7000 | 0.0471        | 0.5108              |
| 2.3962 | 7100 | -             | 0.5167              |
| 2.4300 | 7200 | -             | 0.5165              |
| 2.4637 | 7300 | -             | 0.5149              |
| 2.4975 | 7400 | -             | 0.5184              |
| 2.5312 | 7500 | 0.0465        | 0.5184              |
| 2.5650 | 7600 | -             | 0.5189              |
| 2.5987 | 7700 | -             | 0.5162              |
| 2.6325 | 7800 | -             | 0.5180              |
| 2.6662 | 7900 | -             | 0.5164              |
| 2.7000 | 8000 | 0.0456        | 0.5181              |
| 2.7337 | 8100 | -             | 0.5184              |
| 2.7675 | 8200 | -             | 0.5197              |


### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 4.1.0
- Transformers: 4.52.4
- PyTorch: 2.6.0+cu124
- Accelerate: 1.7.0
- Datasets: 2.14.4
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
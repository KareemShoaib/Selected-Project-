---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:59253
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: lincoln weld machin brand_lincoln
  sentences:
  - lincoln electr bulldog 5500arc stick welder brand_lincoln
  - winchest 100000 btu 80multi-posit ga furnac brand_winchest
  - alexandria mould 34 in. x 3in. x 3in. prime mdf rosett corner block mould brand_alexandria
- source_sentence: circular allen wrench brand_circular
  sentences:
  - huski sae long-arm hex key set 13piec brand_huski
  - south shore furnitur fiesta 23.5in. w microwav kitchen cart with storag on wheel
    in pure black brand_south color_black
  - american standard princeton recess 5ft. left drain bathtub in white brand_american
    color_whit
- source_sentence: acclaim brand_acclaim
  sentences:
  - dewalt 18volt xrp ni-cad cordless reciproc saw kit brand_dewalt
  - 3m bondo 8sq. ft. fiberglass mat brand_3m
  - acclaim light kero collect wall-mount 1light outdoor matt black light fixtur brand_acclaim
    color_black
- source_sentence: mold trim 808501 brand_mold
  sentences:
  - wyndham collect avara 72in. vaniti in espresso with doubl basin stone vaniti top
    in white medicin cabinet brand_wyndham color_whit
  - ornament mould 18251 2in. x 214 in. x 96in. white hardwood emboss ivi bead trim
    chair rail mould brand_ornament color_whit
  - simpli home urban loft 24in. vaniti in espresso brown with quartz marbl vaniti
    top in white under-mount oval sink brand_simpli color_brown color_whit
- source_sentence: clothespin brand_clothespin
  sentences:
  - filament design celesti 3light chrome track light kit with direct head brand_fila
  - westinghous turbo swirl 30in. brush aluminum ceil fan brand_westinghous
  - honey-can-do tradit wood clothespin 96pack brand_honey-can-do
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
      value: 0.5327373057140631
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.5306209932063065
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
    'clothespin brand_clothespin',
    'honey-can-do tradit wood clothespin 96pack brand_honey-can-do',
    'westinghous turbo swirl 30in. brush aluminum ceil fan brand_westinghous',
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
| pearson_cosine      | 0.5327     |
| **spearman_cosine** | **0.5306** |

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

* Size: 59,253 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 10.58 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 25.23 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.69</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                 | sentence_1                                                                                                                           | label                            |
  |:-----------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code>interior door 82inch in length brand_interior</code> | <code>jeld-wen smooth 4panel prime mold interior door slab brand_jeld-wen</code>                                                     | <code>0.665</code>               |
  | <code>cub cadet mower tire brand_cub</code>                | <code>cub cadet xt1enduro seri lt 50in. 24hp v-twin kohler hydrostat ga front-engin ride mower-california compliant brand_cub</code> | <code>0.33499999999999996</code> |
  | <code>granit sealer brand_granit</code>                    | <code>granit gold 24oz. countertop liquid polish brand_granit color_gold</code>                                                      | <code>0.665</code>               |
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
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss | val_spearman_cosine |
|:------:|:-----:|:-------------:|:-------------------:|
| 0.0270 | 100   | -             | 0.3712              |
| 0.0540 | 200   | -             | 0.4146              |
| 0.0810 | 300   | -             | 0.4259              |
| 0.1080 | 400   | -             | 0.4335              |
| 0.1350 | 500   | 0.0624        | 0.4449              |
| 0.1620 | 600   | -             | 0.4453              |
| 0.1890 | 700   | -             | 0.4513              |
| 0.2160 | 800   | -             | 0.4515              |
| 0.2430 | 900   | -             | 0.4561              |
| 0.2700 | 1000  | 0.0583        | 0.4544              |
| 0.2970 | 1100  | -             | 0.4529              |
| 0.3240 | 1200  | -             | 0.4719              |
| 0.3510 | 1300  | -             | 0.4759              |
| 0.3780 | 1400  | -             | 0.4788              |
| 0.4050 | 1500  | 0.0569        | 0.4737              |
| 0.4320 | 1600  | -             | 0.4697              |
| 0.4590 | 1700  | -             | 0.4724              |
| 0.4860 | 1800  | -             | 0.4835              |
| 0.5130 | 1900  | -             | 0.4763              |
| 0.5400 | 2000  | 0.0566        | 0.4801              |
| 0.5670 | 2100  | -             | 0.4826              |
| 0.5940 | 2200  | -             | 0.4849              |
| 0.6210 | 2300  | -             | 0.4882              |
| 0.6479 | 2400  | -             | 0.4837              |
| 0.6749 | 2500  | 0.0552        | 0.4904              |
| 0.7019 | 2600  | -             | 0.4968              |
| 0.7289 | 2700  | -             | 0.4928              |
| 0.7559 | 2800  | -             | 0.4931              |
| 0.7829 | 2900  | -             | 0.4951              |
| 0.8099 | 3000  | 0.0546        | 0.4984              |
| 0.8369 | 3100  | -             | 0.4996              |
| 0.8639 | 3200  | -             | 0.4999              |
| 0.8909 | 3300  | -             | 0.4949              |
| 0.9179 | 3400  | -             | 0.5037              |
| 0.9449 | 3500  | 0.0549        | 0.4966              |
| 0.9719 | 3600  | -             | 0.5039              |
| 0.9989 | 3700  | -             | 0.5022              |
| 1.0    | 3704  | -             | 0.5015              |
| 1.0259 | 3800  | -             | 0.4968              |
| 1.0529 | 3900  | -             | 0.5022              |
| 1.0799 | 4000  | 0.0512        | 0.5092              |
| 1.1069 | 4100  | -             | 0.5062              |
| 1.1339 | 4200  | -             | 0.5042              |
| 1.1609 | 4300  | -             | 0.5038              |
| 1.1879 | 4400  | -             | 0.5032              |
| 1.2149 | 4500  | 0.0497        | 0.5103              |
| 1.2419 | 4600  | -             | 0.5114              |
| 1.2689 | 4700  | -             | 0.5105              |
| 1.2959 | 4800  | -             | 0.5090              |
| 1.3229 | 4900  | -             | 0.5123              |
| 1.3499 | 5000  | 0.0501        | 0.5093              |
| 1.3769 | 5100  | -             | 0.5143              |
| 1.4039 | 5200  | -             | 0.5143              |
| 1.4309 | 5300  | -             | 0.5120              |
| 1.4579 | 5400  | -             | 0.5115              |
| 1.4849 | 5500  | 0.0501        | 0.5123              |
| 1.5119 | 5600  | -             | 0.5098              |
| 1.5389 | 5700  | -             | 0.5165              |
| 1.5659 | 5800  | -             | 0.5169              |
| 1.5929 | 5900  | -             | 0.5154              |
| 1.6199 | 6000  | 0.0497        | 0.5185              |
| 1.6469 | 6100  | -             | 0.5240              |
| 1.6739 | 6200  | -             | 0.5223              |
| 1.7009 | 6300  | -             | 0.5225              |
| 1.7279 | 6400  | -             | 0.5216              |
| 1.7549 | 6500  | 0.0493        | 0.5241              |
| 1.7819 | 6600  | -             | 0.5216              |
| 1.8089 | 6700  | -             | 0.5213              |
| 1.8359 | 6800  | -             | 0.5179              |
| 1.8629 | 6900  | -             | 0.5250              |
| 1.8898 | 7000  | 0.0487        | 0.5261              |
| 1.9168 | 7100  | -             | 0.5229              |
| 1.9438 | 7200  | -             | 0.5209              |
| 1.9708 | 7300  | -             | 0.5217              |
| 1.9978 | 7400  | -             | 0.5269              |
| 2.0    | 7408  | -             | 0.5275              |
| 2.0248 | 7500  | 0.0479        | 0.5271              |
| 2.0518 | 7600  | -             | 0.5255              |
| 2.0788 | 7700  | -             | 0.5189              |
| 2.1058 | 7800  | -             | 0.5241              |
| 2.1328 | 7900  | -             | 0.5262              |
| 2.1598 | 8000  | 0.0459        | 0.5280              |
| 2.1868 | 8100  | -             | 0.5281              |
| 2.2138 | 8200  | -             | 0.5245              |
| 2.2408 | 8300  | -             | 0.5262              |
| 2.2678 | 8400  | -             | 0.5258              |
| 2.2948 | 8500  | 0.0457        | 0.5247              |
| 2.3218 | 8600  | -             | 0.5284              |
| 2.3488 | 8700  | -             | 0.5258              |
| 2.3758 | 8800  | -             | 0.5294              |
| 2.4028 | 8900  | -             | 0.5251              |
| 2.4298 | 9000  | 0.0444        | 0.5262              |
| 2.4568 | 9100  | -             | 0.5286              |
| 2.4838 | 9200  | -             | 0.5283              |
| 2.5108 | 9300  | -             | 0.5272              |
| 2.5378 | 9400  | -             | 0.5284              |
| 2.5648 | 9500  | 0.0455        | 0.5290              |
| 2.5918 | 9600  | -             | 0.5290              |
| 2.6188 | 9700  | -             | 0.5284              |
| 2.6458 | 9800  | -             | 0.5299              |
| 2.6728 | 9900  | -             | 0.5300              |
| 2.6998 | 10000 | 0.0446        | 0.5305              |
| 2.7268 | 10100 | -             | 0.5306              |

</details>

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
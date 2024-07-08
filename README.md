# MarkupLM pre-training
Unofficial implementation of pre-training for [MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding](https://arxiv.org/abs/2110.08518)


## Implementation notes
### Pre-training tasks
- Masked Markup Language Modeling (MMLM)
    - vanilla MLM
    - seem to be "token-level" and not "whole-word-level"
    - **do not mask** tokens in the `<title>`
- Node Relationship Prediction (NRP)   
    - classification of node pairs into set of directed node relationships `{self, parent, child, sibling, ancestor, descendent, others}`
    - using features from the first token of each node only
- Title-Page Matching (TPM)
    - the content within `<title> ... </title>` is randomly replaced with a title from another page
    - given the element `<body>`, predict if the title was replaced using `[CLS]` token fetures
 
### Hyperparameters
- MLM probability = 0.15
- Title-replace probability = 0.15
- Max number of selected pairs in NRP = 1000 per sample
- Ratio of pairs with
`non-others` (i.e., `self`, `parent`, ..) labels = 0.8
- `max_xpath_tag_unit_embeddings` = 256
- `max_xpath_subs_unit_embeddings` = 1024
- `max_depth` = 50
- `pad_width` = 1001
- `xpath_tag_unit_hidden_size` = 32
- `batch_size` = 256
- `num_steps` = 300_000
- `learning rate` = 5e-5
- `warmup_ratio` = 0.06
- `optimizer` = AdamW
- `betas` = (0.9, 0.98)
- `eps` = 1e-6
- `weight decay` = 0.01
- `lr_scheduler` = linear with warmup
- `fp16` = True
- `gradient_checkpointing` = True
- `deepspeed` = unknown config, but enabled
- `backbone` = roberta-base/large

## Notes
- Tokenizer seem to be just Roberta tokenizer with two additional tokens
```
{
  "<end-of-node>": 50266,
  "[empty-title]": 50265
}
```
HTML requires tags `<title>` and `<body>` to be present, but does not require that the content of those tags should be non-empty. Probably, I need to preprocess `html` strings and replace empty titles with `[empty-title]` token. Also, it would make sense not to run TPM on empty titles and on empty bodies.

Note that MarkupLMProcessor will not include tags with empty texts in `xpath_tags_seq`.

```
Some weights of the model checkpoint at microsoft/markuplm-base were not used when initializing MarkupLMModel: ['nrp_cls.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'nrp_cls.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'nrp_cls.decoder.weight', 'ptc_cls.weight', 'ptc_cls.bias', 'nrp_cls.decoder.bias', 'cls.predictions.decoder.weight', 'nrp_cls.LayerNorm.bias', 'nrp_cls.dense.weight', 'cls.predictions.decoder.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing MarkupLMModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing MarkupLMModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```
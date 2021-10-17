# SciCo


This repository contains the data and code for the paper:

[SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts](https://arxiv.org/abs/2104.08809) \
*Arie Cattan, Sophie Johnson, Daniel S. Weld, Ido Dagan, Iz Beltagy, Doug Downey and Tom Hope* \
AKBC 2021. <b>Outstanding Paper Award! 🎉🎉</b>

Check out our [website](https://scico.apps.allenai.org/)!


```
@inproceedings{
    cattan2021scico,
    title={SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts},
    author={Arie Cattan and Sophie Johnson and Daniel S. Weld and Ido Dagan and Iz Beltagy and Doug Downey and Tom Hope},
    booktitle={3rd Conference on Automated Knowledge Base Construction},
    year={2021},
    url={https://openreview.net/forum?id=OFLbgUP04nC}
}
```

**NEW**
- :white_check_mark:&nbsp; Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/longformer-scico).



## Dataset

You can load SciCo directly from [huggingface.co/datasets/allenai/scico](https://huggingface.co/datasets/allenai/scico) as follows:

```python
from datasets import load_dataset
scico = load_dataset("allenai/scico")
```

To download the raw data, click [here](https://nlp.biu.ac.il/~ariecattan/scico/data.tar).

Each file (train, dev, test) is in the `jsonl` format where each row corresponds a topic.
See below the description of the fields in each topic.

* `flatten_tokens`: a single list of all tokens in the topic
* `flatten_mentions`: array of mentions, each mention is represented by [start, end, cluster_id]
* `tokens`: array of paragraphs 
* `doc_ids`: doc_id of each paragraph in `tokens`
* `metadata`: metadata of each doc_id 
* `sentences`: sentences boundaries for each paragraph in `tokens` [start, end]
* `mentions`: array of mentions, each mention is represented by [paragraph_id, start, end, cluster_id]
* `relations`: array of binary relations between cluster_ids [parent, child]
* `id`: id of the topic 
* `hard_10` and `hard_20` (only in the test set): flag for 10% or 20% hardest topics based on Levenshtein similarity.
* `source`: source of this topic PapersWithCode (pwc), hypernym or curated. 


## Model

Our unified model is available on https://huggingface.co/allenai/longformer-scico.
We provide the following code as an example to set the global attention on the special tokens: `<s>`, `<m>` and `</m>`.

 

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-scico')
model = AutoModelForSequenceClassification.from_pretrained('allenai/longformer-scico')

start_token = tokenizer.convert_tokens_to_ids("<m>")
end_token = tokenizer.convert_tokens_to_ids("</m>")

def get_global_attention(input_ids):
    global_attention_mask = torch.zeros(input_ids.shape)
    global_attention_mask[:, 0] = 1  # global attention to the CLS token
    start = torch.nonzero(input_ids == start_token) # global attention to the <m> token
    end = torch.nonzero(input_ids == end_token) # global attention to the </m> token
    globs = torch.cat((start, end))
    value = torch.ones(globs.shape[0])
    global_attention_mask.index_put_(tuple(globs.t()), value)
    return global_attention_mask
    
m1 = "In this paper we present the results of an experiment in <m> automatic concept and definition extraction </m> from written sources of law using relatively simple natural methods."
m2 = "This task is important since many natural language processing (NLP) problems, such as <m> information extraction </m>, summarization and dialogue."

inputs = m1 + " </s></s> " + m2  

tokens = tokenizer(inputs, return_tensors='pt')
global_attention_mask = get_global_attention(tokens['input_ids'])

with torch.no_grad():
    output = model(tokens['input_ids'], tokens['attention_mask'], global_attention_mask)
    
scores = torch.softmax(output.logits, dim=-1)
# tensor([[0.0818, 0.0023, 0.0019, 0.9139]]) -- m1 is a child of m2
```


**Note:** There is a slight difference between this model and the original model presented in the [paper](https://openreview.net/forum?id=OFLbgUP04nC). 
The original model includes a single linear layer on top of the `<s>` token (equivalent to `[CLS]`) 
while this model includes a two-layers MLP to be in line with `LongformerForSequenceClassification`.   
You can download the original model as follows:
```python
curl -L -o model.tar https://www.dropbox.com/s/cpcnpov4liwuyd4/model.tar?dl=0
tar -xvf model.tar 
rm model.tar 
```


## Training and Evaluation 

### Getting started:

You may wish to create a conda environment:
```
conda create --name scico python=3.8
conda activate scico 
```
 
Install all dependencies using `pip install -r requirements.txt`. \
We provide the code for training the baseline models, Pipeline and Multiclass.



### Baseline

The baseline model uses our recent cross-document coreference model [(Cattan et al., 2021)](https://aclanthology.org/2021.findings-acl.453.pdf), 
the code is in [this](https://github.com/ariecattan/coref) repo.

* __Training__: Set the `configs/config_pairwise.json` file: select any BERT model to run in the field `bert_model` and set the directory to save the model in `model_path`.
This script will save a model at each epoch. 

```
python train_cd_coref_scorer.py --config configs/config_pairwise.json
```

* __Fine-tuning threshold__: (1) Run inference on the dev set using all the saved models and different values of thresholds 
and (2) run the scorer on the above predictions to get the best model with the best threshold. Make sure to set `data_path` 
to the dev set path, `bert_model` to the corresponding BERT model, and `save_path` to 
the corresponding directory to save the conll files.

```
python tune_coref_threshold.py --config configs/config_clustering_cs_roberta
python run_coref_scorer [folder_dev_pred] [gold_dev_conll]
```

* __Inference__: Run inference on the test test, make sure to set the `data_path` 
to the test set path.  You also need to set the name of an `nli_model` in the config 
for predicting the relations between the clusters. 
```
python predict_cd_coref_entailment.py --config configs/config_clustering_cs_roberta
```

* __Inference (cosine similarity)__: We also provide a script for clustering the 
mentions using an agglomerative clustering only on cosine similarity between
the average-pooling of the mentions. Relations between clusters are also predicted using an entailment 
model. 
```
python predict_cosine_similarity.py --gpu 0 \
    --data_path data/test.jsonl \
    --output_dir checkpoints/cosine_roberta \
    --bert_model roberta-large \
    --nli_model roberta-large-mnli \
    --threshold 0.5 
``` 


### Cross-Encoder pipeline and Multiclass

For both training and inference, running the pipeline or the multiclass model
can be done with only modifying the args `--multiclass` to {pipeline, multiclass}.


* __Training__:  Set important config for the model and data path in `configs/multiclass.yaml`,
then run the following script: 
```
python train.py --config configs/multiclass.yaml \
    --multiclass multiclass # (or coref or hypernym) 
```
  

* __Fine tuning threshold__: After training the multiclass model, we need to tune on the dev set 
the threshold for the agglomerative clustering and the stopping criterion for the 
hierarchical relations. 

```
python tune_hp_multiclass.py --config configs/multiclass.yaml 
```


* __Inference__: Set the path to the checkpoints of the models and the best thresholds, run
the following script on the test set.

```
python predict.py --config configs/multiclass.yaml \
    --multiclass multiclass # (or pipeline) 
```


### Evaluation 

Each inference script produces a `jsonl` file with the fields `tokens`, `mentions`, `relations` and `id`.
Models are evaluated using the usual coreference metrics using the [coval](https://github.com/ns-moosavi/coval/) script,
 hierarchy (recall, precision and F1), and directed path ratio. 

```
python evaluate.py [gold_jsonl_path] [sys_jsonl_path] options
```

If you want to evaluate only on the hard topics (based on levenshtein performance, see Section 4.5), 
you can set the `options` to be `hard_10`, `hard_20` or `curated`.

# SciCo


This repository contains the data and code for the paper:

[SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts](https://arxiv.org/abs/2104.08809) \
*Arie Cattan, Sophie Johnson, Daniel S. Weld, Ido Dagan, Iz Beltagy, Doug Downey and Tom Hope* \
2021 

Check out our [website](https://scico.apps.allenai.org/)!


```
@article{Cattan2021SciCoHC,
    title={SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts},
    author={Arie Cattan and Sophie Johnson and Daniel S. Weld and Ido Dagan and Iz Beltagy and Doug Downey and Tom Hope},
    journal={ArXiv},
    year={2021},
    volume={abs/2104.08809}
}
```



## Dataset



Click [here](https://nlp.biu.ac.il/~ariecattan/scico/data.tar) to download SciCo.

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


## Models and Evaluation 

### Getting started:

You may wish to create a conda environment:
```
conda create --name scico python=3.8
conda activate scico 
```
 
Install all dependencies using `pip install -r requirements.txt`. \
Install `pytorch lightning` from source in [this](https://github.com/PyTorchLightning/pytorch-lightning/) repo.


We provide the code for training the baseline models, Pipeline and Multiclass.



### Baseline

The baseline model uses our recent cross-document coreference model [(Cattan et al., 2020)](https://arxiv.org/abs/2009.11032), 
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
    --method cross
    --multiclass multiclass # (or pipeline) 
```
  

* __Fine tuning threshold__: After training the multiclass model, we need to tune on the dev set 
the threshold for the agglomerative clustering and the stopping criterion for the 
hierarchical relations. 

```
python tune_hp_multiclass.py --config configs/multiclass.yaml
    --method cross
```


* __Inference__: Set the path to the checkpoints of the models and the best thresholds, run
the following script on the test set.

```
python predict.py --config configs/multiclass.yaml \
    --method cross
    --multiclass multiclass # (or pipeline) 
```


### Evaluation 

Each inference script produces a `jsonl` file with the fields `tokens`, `mentions`, `relations` and `id`.
Models are evaluated using the usual coreference metrics using the [coval](https://github.com/ns-moosavi/coval/) script,
 hierarchy (recall, precision and F1), directed path ratio and AUC on the multiclass. 

```
python evaluate.py [gold_jsonl_path] [sys_jsonl_path]
```


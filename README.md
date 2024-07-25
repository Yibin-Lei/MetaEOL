# Code Implementation for MetaEOL
## Python Env

```
conda create -n metaeol_39 python=3.9
conda activate metaeol_39

pip install -r requirements.txt
```

## Download data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
```

## Running Script
```
python evaluation.py --model_name_or_path "mistralai/Mistral-7B-v0.1" --mode test --task_set sts --prompt_method prompteol
```
The argument `task_set` can also be set to `transfer`. Similarly, the argument `prompt_method` can also be set to `metaeol`.

## Acknowledgement

Our code is developed upon [PromptEOL](https://github.com/kongds/scaling_sentemb). We thank the authors of PromptEOL for their great efforts.
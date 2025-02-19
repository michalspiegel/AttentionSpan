# Benchmarking Attention on Algorithmic Reasoning

The main purpose of this repository is to provide utilities for generating training and evaluation data for various Algoritmic Reasoning tasks. However, we also provide workflow for evaluating arbitrary transformer-based model and inspecting the results, in particular, the attention scores.

## How do I generate the data?

Each task is implemented as a configurable generator that can be used in the following manner:
```python
from generators.long_addition import LongAdditionGenerator
# Instantiate the generator with a tokenizer and other configurable parameters of the task
data_generator = LongAdditionGenerator(tokenizer=tokenizer, **other_parameters)
# Generate 100 samples
samples = itertools.islice(data_generator.generate_samples(), 100)
# Convert the sample into a Pytorch Dataset that can be easily used for training.
dataset = data_generator.generate_dataset(sample) 
```
In this example, we instantiated a generator for the long addition task, we generated 100 samples and turned them into a Pytorch Dataset.

## How to evaluate a model?

### HuggingFace Transformers

To train and evaluate a model available from the HuggingFace Transformers library, use an already provided script *evaluate/evaluate_transformers_model.py*. The script expects a configuration file, see examples in *evaluate/configs*. Afterwards just execute the script with the particular config file:
```bash
python evaluate_transformers_model.py --config-name config_filepath.yaml
```
Optionally, Hydra enables you to override specific configurations directly in the command line. For example, to disable training, just execute:
```bash
python evaluate_transformers_model.py --config-name config_filepath.yaml training=null
```
Or change the batch size using:
```bash
python evaluate_transformers_model.py --config-name config_filepath.yaml training.batch_size=32
```
### Other
Evaluating other models will require editing the provided evaluation script. If it is a model implemented in Pytorch, then it should be enough to change just the model loading logic.

## Attention Analysis

This repository also implements some tools for analyzing attention scores, the code can be found in module *attention_evaluation*. In particular, we provide some analysis on aggregated attention scores in `attention_evaluation/aggregation.py` (using attention rollout) and visualization of aggregated attention matrices in `attention_evaluation/visualization.py`.
All of these analysis tool are currently used when calling the `evaluate/evaluate_transformers_model.py`, mainly:
- mean proportion of attention score on reference tokens
    - This tries to estimate how much the model attends the tokens we expect him to attend
- mean proportion attention score on reference tokens given correct/incorrect prediction
    - These 2 scores try to estimate whether there's a significant change in attention reference tokens when the model makes incorrect prediction (vs. correct prediction)
- statistical tests on these means to estimate the statistical significance 


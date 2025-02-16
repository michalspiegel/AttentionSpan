wandb offline

# Hyperparameter search on reversal and addition
python evaluate_transformers_model.py \
--multirun \
--config-name llama_instruct_string_reversal.yaml \
model.model_configuration.attention_probs_dropout_prob='interval(0,0.2)' \
model.model_configuration.hidden_dropout_prob='interval(0,0.2)' \
training.batch_size='choice(4, 16, 32, 64)' \
training.learning_rate='interval(1e-6,1e-4)' \
training.optimizer_parameters.beta1='interval(0.9,0.99)' \
training.optimizer_parameters.beta2='interval(0.99,0.99)' \
training.optimizer_parameters.weight_decay='interval(0.01,0.5)' \
training.dataset.num_samples=1000 \
evaluation.batch_size=16 \
evaluation.dataset1.num_samples=16 \
evaluation.dataset2.num_samples=100 \
training.save_checkpoint=False \
evaluation.visualize=False \
hydra/sweeper=optuna \
hydra.sweeper.study_name=accuracy \
hydra.sweeper.direction=maximize \
hydra.sweeper.n_trials=20 \
hydra.sweeper.n_jobs=1 


# Hyperparameter search on reversal and addition
python evaluate_transformers_model.py \
--multirun \
--config-name llama_instruct_addition.yaml \
model.model_configuration.attention_probs_dropout_prob='interval(0,0.2)' \
model.model_configuration.hidden_dropout_prob='interval(0,0.2)' \
training.batch_size='choice(4,16,32)' \
training.learning_rate='interval(1e-6,1e-4)' \
training.optimizer_parameters.beta1='interval(0.9,0.99)' \
training.optimizer_parameters.beta2='interval(0.99,0.99)' \
training.optimizer_parameters.weight_decay='interval(0.01,0.5)' \
training.dataset.num_samples=1000 \
evaluation.batch_size=16 \
evaluation.dataset1.num_samples=16 \
evaluation.dataset2.num_samples=100 \
training.save_checkpoint=False \
evaluation.visualize=False \
hydra/sweeper=optuna \
hydra.sweeper.study_name=accuracy \
hydra.sweeper.direction=maximize \
hydra.sweeper.n_trials=20 \
hydra.sweeper.n_jobs=1 
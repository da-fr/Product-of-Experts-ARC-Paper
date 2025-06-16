# Copyright 2024-2025 Daniel Franzen, Jan Disselhoff and David Hartmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import bz2
import pickle
from datasets import Dataset
from diskcache import Cache

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator, load_unsloth_model, save_model_and_tokenizer
from inference_tools import inference_run
from selection import EvalTool


# input paths
base_model = 'da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit'  # auto-downloaded from huggingface.co
path_to_sudoku_3m = os.path.join('input', 'sudoku', 'sudoku-3m.csv')
# please download dataset from https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings

# output paths
output_path = 'output_evaluation_LlamaReArc_on_Sudoku3m'

# load sudoku dataset
train_dataset = ArcDataset.load_sudoku_csv(path_to_sudoku_3m,
    num_challenges=128000, diff_lim=None, examples_per_challenge=8, max_indx=1200000)
eval_dataset = ArcDataset.load_sudoku_csv(path_to_sudoku_3m,  # use indx starting at 2 million for eval
    num_challenges=1000, diff_lim=None, diff_lim_lower=None, examples_per_challenge=1, min_indx=2000000, max_indx=3000000)

# load model
save_model_path = os.path.join(output_path, 'finetuned_model')
retrain = not os.path.exists(save_model_path)
model, tokenizer = load_unsloth_model(base_model if retrain else save_model_path, bits=4)

# set formatting options
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=128000,
)

if retrain:
    # create lora model
    lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                   'embed_tokens', 'lm_head']
    model = FastLanguageModel.get_peft_model(
        model=model,
        target_modules=lora_layers,
        r=256,
        lora_alpha=24,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    # run finetuning
    FastLanguageModel.for_training(model)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(train_dataset.as_list(len_name='text', **fmt_opts)),
        eval_dataset=Dataset.from_list(eval_dataset.as_list(len_name='text', **fmt_opts)),
        dataset_text_field="text",
        max_seq_length=fmt_opts['max_tokens'],
        data_collator=InputMaskingDataCollator(
            instruction_template=fmt_opts['query_beg'],
            response_template=fmt_opts['reply_beg'],
            mlm=False,
            tokenizer=tokenizer,
            mask_first_n_examples=0,
        ),
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_ratio=0.25,
            num_train_epochs=1,
            learning_rate=1e-4,
            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.00,
            lr_scheduler_type='cosine',
            seed=42,
            output_dir='tmp_output',
            save_strategy='no',
            report_to='none',
        ),

    )
    trainer_stats = unsloth_train(trainer)
    save_model_and_tokenizer(save_model_path, model, tokenizer)

# run inference
FastLanguageModel.for_inference(model)
infer_aug_opts = dict(tp='all', rt='all', perm=True, keep_bg=True, shfl_ex=True, seed=10000)
infer_dataset = eval_dataset.augment(**infer_aug_opts)

inference_keys = {}
for k in infer_dataset.keys:
    base_key = k.split('.', 1)[0]
    if base_key not in inference_keys:
        inference_keys[base_key] = []
    inference_keys[base_key].append(k)

eval_tool = EvalTool(n_guesses=1)  #only 1 guess in sudoku
inference_results = inference_run(
    model_tok=(model, tokenizer),
    fmt_opts=fmt_opts,
    dataset=infer_dataset,
    min_prob=0.01,
    aug_score_opts=infer_aug_opts,
    callback=eval_tool.process_result,
    cache=Cache(os.path.join(output_path, 'inference_cache')).memoize(typed=True, ignore=set(['model_tok', 'guess'])),
)

# dump results
results_file = os.path.join(output_path, 'results'+task_name+'.pickle.bz2')
with bz2.BZ2File(results_file, 'w') as f:
    pickle.dump(inference_keys, f)
    pickle.dump(inference_results, f)

# write submission
submission_file = os.path.join(output_path, 'submission.json')
with open(submission_file, 'w') as f:
    json.dump(eval_dataset.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))

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
from tqdm import tqdm
from datasets import Dataset
from diskcache import Cache

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator, load_unsloth_model
from inference_tools import inference_run
from selection import EvalTool
from arc_downloader import download_arc_data

# configuration
base_model = 'da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit'  # auto-downloaded from huggingface.co
arc_data_path = os.path.join('input', 'arc-prize-2024')  # format as on kaggle, auto-downloaded from arc git
output_path = 'output_evaluate_LlamaReArc_on_ARCeval_16x_dfs0.090_with_ttt'
ttt_target_size = 1

# load evaluation dataset
download_arc_data(arc_data_path)
eval_dataset = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
eval_dataset = eval_dataset.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))

# setup formatting options
tokenizer = load_unsloth_model(base_model, bits=4)[1]
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=16384,
)

# main loop
inference_keys = {}
inference_results = {}
eval_tool = EvalTool(n_guesses=2)

with tqdm(eval_dataset.split(n=len(eval_dataset.challenge)//ttt_target_size, split_seed=123), desc='inference') as pbar:
    for i, eval_dataset_part in enumerate(pbar):

        # test-time training
        def get_model_and_tokenizer(cache=[None]):
            if cache[0] is None:
                # load base model
                model, tokenizer = load_unsloth_model(base_model, bits=4)

                # create lora model
                lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                               'embed_tokens', 'lm_head']
                model = FastLanguageModel.get_peft_model(
                    model=model,
                    target_modules=lora_layers,
                    r=32,
                    lora_alpha=16,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=True,
                    loftq_config=None,
                )

                # augment training data
                train_aug_opts = dict(tp='all', rt='all', shfl_keys=True, perm=True, shfl_ex=True, seed=i)
                train_dataset_aug = eval_dataset_part.remove_test_data().augment(n=8, **train_aug_opts)
                train_dataset_as_list = train_dataset_aug.as_list(len_name='text', **fmt_opts)

                # run training process
                FastLanguageModel.for_training(model)
                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=Dataset.from_list(train_dataset_as_list),
                    dataset_text_field="text",
                    max_seq_length=fmt_opts['max_tokens'],
                    data_collator=InputMaskingDataCollator(
                        instruction_template=fmt_opts['query_beg'],
                        response_template=fmt_opts['reply_beg'],
                        mlm=False,
                        tokenizer=tokenizer,
                        mask_first_n_examples=1,
                    ),
                    args=TrainingArguments(
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=1,
                        warmup_steps=32,
                        num_train_epochs=1,
                        learning_rate=1e-4,
                        embedding_learning_rate=1e-5,
                        fp16=not is_bfloat16_supported(),
                        bf16=is_bfloat16_supported(),
                        logging_steps=8,
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
                FastLanguageModel.for_inference(model)
                cache[0] = (model, tokenizer)
            return cache[0]

        # run inference
        task_cache_path = os.path.join(output_path, f'{sorted(eval_dataset_part.challenge.keys())[0]}.cache')
        infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000 + i)
        eval_dataset_augmented = eval_dataset_part.augment(n=2, **infer_aug_opts)

        for k in eval_dataset_augmented.keys:
            base_key = k.split('.', 1)[0]
            if base_key not in inference_keys:
                inference_keys[base_key] = []
            inference_keys[base_key].append(k)

        inference_results.update(
            inference_run(
                model_tok=get_model_and_tokenizer,
                fmt_opts=fmt_opts,
                dataset=eval_dataset_augmented,
                min_prob=0.09,
                aug_score_opts=dict(n=2, **infer_aug_opts),
                callback=eval_tool.process_result,
                cache=Cache(task_cache_path).memoize(typed=True, ignore=set(['model_tok', 'guess'])),
                print_func=pbar.write,
            )
        )

# dump results
results_file = os.path.join(output_path, 'results.pickle.bz2')
with bz2.BZ2File(results_file, 'w') as f:
    pickle.dump(inference_keys, f)
    pickle.dump(inference_results, f)

# write submission
submission_file = os.path.join(output_path, 'submission.json')
with open(submission_file, 'w') as f:
    json.dump(eval_dataset.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Reload score for '{submission_file}':", eval_dataset.validate_submission(json.load(f)))

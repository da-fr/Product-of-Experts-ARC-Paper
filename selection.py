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

import numpy as np


def max_gen_prob(res):
    return min(res['scores_inf'].values())


def max_aug_prob(res):
    return min(res['scores_aug'].values())


def min_aug_prob(res):
    return max(res['scores_aug'].values())


def sum_aug_prob(res):
    return sum([-np.exp(-s) for s in res['scores_aug'].values()])


def mul_aug_prob(res):
    return sum(res['scores_aug'].values())


all_score_algos = [
    max_gen_prob,  # highest probability from inference results
    max_aug_prob,  # highest probability from augmented scoring
    min_aug_prob,  # lowest probability from augmented scoring
    sum_aug_prob,  # sum of probabilities from augmented scoring
    mul_aug_prob,  # sum of log probabilities from augmented scoring
]


class EvalTool(object):   # providing on-the-fly evaluation of scoring algorithms
    def __init__(self, n_guesses, score_algos=all_score_algos, sorting_algo='mul_aug_prob', use_not_found=False):
        self.score_algos = score_algos
        self.n_guesses = n_guesses  # number of guesses allowed
        self.sorting_algo = sorting_algo  # sorting algorithm for results, relevant for final submission (default: last)
        self.n_acc = [0] * len(score_algos)  # counting correct n-guesses for different scoring algorithms
        self.a_acc = 0  # counting cases where the solution is found at all
        self.count = 0  # counting number of tasks seen
        self.use_not_found = use_not_found

    def process_result(self, res, name, value, print_func=print):
        for r in res:
            r['scores_alg'] = [algo(r) for algo in self.score_algos]
        if self.sorting_algo is not None:
            res.sort(key=lambda x: x['scores_alg'][[a.__name__ for a in self.score_algos].index(self.sorting_algo)])
        if not self.use_not_found:
            res = [r for r in res if r['scores_inf']]
        pos = ([i for i, r in enumerate(res) if r['correct']] + [None])[0]
        self.count += value
        self.a_acc += value if pos is not None else 0
        corr_info = f"{len(res)} candidates, correct solution {'not found' if pos is None else 'FOUND'}"
        if print_func is not None:
            print_func(f" * task '{name}': {corr_info}")
        for i, algo in enumerate(self.score_algos):
            if pos is not None:
                scores = [r['scores_alg'][i] for r in res]
                rank = np.argsort(np.argsort(scores))[pos]
                if rank < self.n_guesses:
                    self.n_acc[i] += value
            rank_info = f", corr_sol. @{rank + 1:>2} / {len(res)}" if pos is not None else ''
            n_acc_info = f"{self.n_acc[i] / self.count:7.2%} ({self.n_acc[i]:>6.2f}/{self.count:>6.2f})"
            if print_func is not None:
                print_func(f"   {f'{self.score_algos[i].__name__}:':14} {n_acc_info}{rank_info}")
        a_acc_info = f"{self.a_acc / self.count:7.2%} ({self.a_acc:>6.2f}/{self.count:>6.2f})"
        if print_func is not None:
            print_func(f"   {'correct_found:':14} {a_acc_info}\n")

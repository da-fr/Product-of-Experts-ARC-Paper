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

import requests
import zipfile
import os
import io
import re
import json


zip_url = 'https://codeload.github.com/fchollet/ARC-AGI/zip/refs/heads/master'
subset_names = ['training', 'evaluation']


def download_arc_data(arc_data_path):
    # check if files are already there
    required_files = []
    for subset in subset_names:
        required_files.append(os.path.join(arc_data_path, f'arc-agi_{subset}_challenges.json'))
        required_files.append(os.path.join(arc_data_path, f'arc-agi_{subset}_solutions.json'))
    if all(map(os.path.isfile, required_files)): return

    # download repo
    r = requests.get(zip_url)
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # extract subsets
    extract_id = re.compile('^ARC-AGI-master/data/([a-z]+)/([a-z0-9]+)[.]json')
    datasets = {}
    for f in z.filelist:
        id = extract_id.match(f.filename)
        if id:
            if id.group(1) not in datasets: datasets[id.group(1)] = {}
            datasets[id.group(1)][id.group(2)] = json.loads(z.read(f))

    # store challenges and solutions seperately
    os.makedirs(arc_data_path, exist_ok=True)
    for subset, challenges in datasets.items():
        solutions = {}
        for k, v in challenges.items():
            assert v.pop('name', k) == k  # remove name tags that occur inconsistently in the data
            solutions[k] = [t.pop('output') for t in v['test']]
        with open(os.path.join(arc_data_path, f'arc-agi_{subset}_challenges.json'), 'w') as f: json.dump(challenges, f)
        with open(os.path.join(arc_data_path, f'arc-agi_{subset}_solutions.json'), 'w') as f: json.dump(solutions, f)
        print(f'Downloaded arc {subset} set.')

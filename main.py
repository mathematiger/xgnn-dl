# This File does the experiments and saves them to content/
import torch_geometric
import torch
import numpy as np
import random
from beamsearch import BeamSearch
from models import GNNDatasets
from syntheticdatasets import SyntheticDatasets
from evaluation import FidelityEvaluator
from main_utils import create_gnn_and_dataset, create_test_dataset, DualLogger
from owlapy.render import DLSyntaxObjectRenderer
import os
import sys
from datetime import datetime
dlsr = DLSyntaxObjectRenderer()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example of setting the seed
set_seed(1)


# Save the out-logs to the folder content/Results
output_dir = "content/Results"
os.makedirs(output_dir, exist_ok=True)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"output_log_{current_datetime}.txt"
log_file_path = os.path.join(output_dir, log_file_name)
sys.stdout = DualLogger(log_file_path)
# Clean up old log files, keeping only the last 10
DualLogger.cleanup_old_logs(output_dir, keep=10)


# setup for parameters

retrain_GNN_and_data = True
datasets = {'house': 'house',
            'circle': 'circle',
            'star': 'star',
            'wheel': 'wheel',
            'mutag': {'positive': ['mutagNO2', 'mutagNH2'], 'negative': ['circle']},
            'combi': {'positive': ['house', 'wheel'], 'negative': ['circle',  'star']},
            'mutagshort': {'positive': ['NO2', 'NH2'], 'negative': []},
            }

datasets = {
    'mutag': {'positive': ['mutagNO2', 'mutagNH2'], 'negative': ['circle']},
    'combi': {'positive': ['house', 'wheel'], 'negative': ['circle',  'star']},
    'mutagshort': {'positive': ['NO2', 'NH2'], 'negative': []},
}

dict_types_to_classify = {'house': 'A',
                          'circle': 'A', 'star': 'A', 'wheel': 'A', 'combi': 'A', 'mutag': 'A', 'mutagshort': 'A'}
gnn_parameters = [{'name': 'SAGE_2_100', 'gnn_layers': 4, 'epochs': 100},
                  ]


# ------- Code -----------
gnns = dict()
data = dict()
data_cl = dict()

for ds in datasets.keys():
    for gnnparams in gnn_parameters:
        gnns[ds], data[ds], data_cl[ds] = create_gnn_and_dataset(dataset_name=ds,
                                                                 dataset=datasets[ds],
                                                                 gnn_name=gnnparams['name'],
                                                                 gnn_epochs=gnnparams['epochs'],
                                                                 gnn_layers=gnnparams['gnn_layers'],
                                                                 type_to_classify='A',  # default: A
                                                                 retrain=retrain_GNN_and_data,
                                                                 num_nodes=1000,
                                                                 )

        # beam search CEs
        beam_search = BeamSearch(gnns[ds].model,
                                 data[ds],
                                 beam_width=200,
                                 beam_depth=12,
                                 # max_depth of created CEs, should be number of GNN layers
                                 max_depth=gnnparams['gnn_layers'],
                                 )
        beam = beam_search.beam_search()

        # return top CEs
        test_ce_dataset = create_test_dataset(
            dataset_name=ds, num_nodes=1000, dataset=datasets[ds])
        fideval = FidelityEvaluator(
            test_ce_dataset, gnns[ds].model, type_to_explain=dict_types_to_classify[ds])
        print(f"Top 10 CEs: of dataset {ds} and gnn {gnnparams['name']}")
        for i in range(10):

            acc = fideval.score_fid_accuracy(beam[i])
            print(
                f"Number {i+1} CE is {dlsr.render(beam[i])} and has an acc of {round(acc, 2)}")

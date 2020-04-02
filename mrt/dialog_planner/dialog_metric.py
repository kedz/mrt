from plum2.metrics import PlumMetric
import numpy as np
from scipy import stats
from collections import defaultdict


class DialogMetric(PlumMetric):
    def __init__(self, output_getter, target_getter, da_getter):
        super(DialogMetric, self).__init__()
        self.output_getter = output_getter
        self.target_getter = target_getter
        self.da_getter = da_getter

    def update(self, model, batch, forward_state):
        
        output = self.output_getter(model, batch, forward_state)
        target = self.target_getter(model, batch, forward_state)
        das = self.da_getter(model, batch, forward_state)
        for pred_y, true_y, da in zip(output, target, das):
            if set(pred_y) != set(true_y):
                self.kts.append(0)
                continue

            if len(pred_y) != len(true_y):
                self.kts.append(0)
                continue

            if len(true_y) == 1:
                continue

            rank_map = {yi: i for i, yi in enumerate(pred_y)}
            true_ranks = [rank_map[yi] for yi in true_y]
            pred_ranks = [rank_map[yi] for yi in pred_y]
            kt, _ = stats.kendalltau(true_ranks, pred_ranks)
            self.kts.append(kt)
            self.da_kts[da].append(kt)

    def apply_metric(self):
        avg_kt = np.mean(self.kts)
        r = {
            "kendall's_tau": {
                "MEAN": avg_kt,
            }
        }
        for da, kts in self.da_kts.items():
            r["kendall's_tau"][da] = np.mean(kts)
        return r

    @property
    def min_criterion(self):
        return False

    @property
    def scalar(self):
        return self.compute()["kendall's_tau"]['MEAN']

    def reset_stats(self):
        self.kts = []
        self.da_kts = defaultdict(list)




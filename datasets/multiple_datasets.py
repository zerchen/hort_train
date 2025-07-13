import random
import numpy as np
from torch.utils.data.dataset import Dataset

class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.min_db_data_num = min([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.min_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.min_db_data_num
            data_margin = len(self.dbs[db_idx]) - self.min_db_data_num
            if data_margin > 0:
                db_idx_margin = random.randint(0, data_margin)
            else:
                db_idx_margin = 0
            data_idx = (index - db_idx * self.min_db_data_num) % len(self.dbs[db_idx]) + db_idx_margin
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]

import json
import os.path
import numpy as np

class Simulator:
    def __init__(self, mode):
        self.tuple_dict = {}
        self.all_apis = []
        with open(os.path.join('../raw_data/all', f'{mode}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                interaction = line.strip('\n').split(' ')
                interaction = list(map(int, interaction))
                if interaction[0] not in self.tuple_dict:
                    self.tuple_dict[interaction[0]] = interaction[1:]
                    for a in interaction[1:]:
                        if a not in self.all_apis:
                            self.all_apis.append(a)
        f.close()
        self.mashup_ids = list(self.tuple_dict.keys())
        self.num_mashups = len(self.mashup_ids)
        self.all_apis = sorted(self.all_apis)

    def __len__(self):
        return len(self.tuple_dict)

    def get_data(self, mashup_idx):
        mashup_id = self.mashup_ids[mashup_idx % self.num_mashups]
        api_ids = np.array(self.tuple_dict[mashup_id])
        all_simulator_apis = self.all_apis
        return mashup_id, api_ids, all_simulator_apis

    def step(self, state, recommended_api_ids, n_round):  # mean_api_nums_for_each_mashup : 3.3366141732283463
        mashup_id = state[0][0]
        selected_api_ids = state[1]
        all_api_ids = self.tuple_dict[mashup_id]
        remain_api_ids = list(set(all_api_ids) - set(selected_api_ids))
        dcg = 0.0
        idcg = np.sum(1.0 / np.log2(np.arange(2, len(recommended_api_ids) + 2)))
        for i,r in enumerate(recommended_api_ids):
            if r in remain_api_ids:
                dcg += 1.0 / np.log2(i + 2)
        r1 = dcg / idcg
        if set(all_api_ids) == set(selected_api_ids):
            r2 = 0
        else:
            r2 = -0.01 * n_round
        return r1 + r2


if __name__ == '__main__':
    simulator = Simulator('test')
    num_users = len(simulator)
    for u in range(num_users):
        mashup_id, api_ids, all_simulator_apis = simulator.get_data(u)
        copy_all = all_simulator_apis.copy()
        print(copy_all)
    '''mashup_id, api_ids = simulator.get_data(12)
    api_ids = np.append(api_ids,-1)
    print(mashup_id, api_ids)
    for api in api_ids:
        if api != -1:
            print(api)'''
    '''
    reward = simulator.step(23, [3556, 3559, 3560], [3577, 3525, 3592])
    print(reward)
    list1 = [[1,2,3],[1,2]]
    list2 = [5,6,7,8,1,9,2]
    print(torch.stack([torch.tensor(s) for s in list1]))'''
import os
import numpy as np
import random
import time
import json
import torch

dataset = ['Instruments', 'Beauty', 'Yelp', 'Games', 'Arts'][2]
data_dir = '../../data/'+ dataset +'/'

num_of_users = 10

if not os.path.exists(dataset):
    os.mkdir(dataset)

embedding_dict = torch.load(os.path.join(data_dir + 'embedding_dict_512.pth'))
print(embedding_dict)

user_embs = embedding_dict['user_emb'].detach().cpu()
item_embs = embedding_dict['item_emb'].detach().cpu()
nuser = len(user_embs)
nitem = len(item_embs)

item_tokens = []
special_token_2_index = {}
new_tokens = set()
user_traj = {}
item_traj = {}
with open(os.path.join(data_dir + dataset + ".inter.json"), 'r') as f:
    inters = json.load(f)
    f.close()

    for inter in inters:
        user_id = int(inter)
        if user_id not in user_traj: user_traj[user_id] = []
        for item_id in inters[inter]:
            if int(item_id) not in item_traj:
                item_traj[int(item_id)] = []
            user_traj[int(user_id)].append(int(item_id))
            item_traj[int(item_id)].append(int(user_id))

import tsne

for i in range(60):
    user_idx = random.sample(range(1, nuser), (num_of_users))
    random.shuffle(user_idx)

    item_idx, labels = [], []
    index_ = 0
    for i, user_id in enumerate(user_idx):
        item_idx.extend(user_traj[user_id])
        labels.extend([index_ for _ in range(len(user_traj[user_id]))] )
        index_ += 1

    item_idx, labels = np.array(item_idx), np.array(labels)
    print('#User:', len(user_idx), '#item:', len(item_idx))
    X = np.array(item_embs[item_idx])
    Y = np.array(labels)

    time_ = str(time.time()).split('.')[0]
    tsne.visualization(X, Y, dataset + '/Item_' + time_)


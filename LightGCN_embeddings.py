import torch
import os
import json

data_name = ['Instruments', 'Beauty', 'Yelp', 'Games', 'Arts'][2]
data_dir = './data/'+ data_name +'/'
top_k = 10

embedding_dict = torch.load(os.path.join(data_dir + 'embedding_dict.pth'))
print(embedding_dict)
user_embs = embedding_dict['user_emb'].detach()
item_embs = embedding_dict['item_emb'].detach()

item_tokens = []
special_token_2_index = {}
new_tokens = set()
with open(os.path.join(data_dir + data_name + ".inter.json"), 'r') as f:
    inters = json.load(f)
    f.close()
with open(os.path.join(data_dir + data_name + ".index.json"), 'r') as f:
    indices = json.load(f)
    f.close()
    for index in indices.values():
        for token in index:
            new_tokens.add(token)
    new_tokens = sorted(list(new_tokens))
    for i, token in enumerate(new_tokens):
        special_token_2_index[token] = i

nuser, nitem, ntoken = len(inters), len(indices), len(new_tokens)
print("Dataset:", data_name)
print('#user', nuser, '#item', nitem, '#total', (nuser + nitem))

logists = {}
for user_id in inters:
    user_emb = user_embs[int(user_id)]
    scores = torch.tensor(torch.matmul(user_emb, item_embs.T), dtype=torch.float16)

    # For Yelp, AGRec will benefit from interacted items in the past.
    # if data_name != 'Yelp':
    histories = inters[user_id][:-1]
    scores[list(histories)] = -1e4

    topk_items = list(torch.topk(scores, k=top_k)[1].cpu().numpy())

    # =========== Generate FSMs, consisting of four states.
    logists[user_id] = {}
    for item_id in topk_items[::-1]:
        a, b, c, d = indices[str(item_id)]
        a, b, c, d = special_token_2_index[a], special_token_2_index[b], special_token_2_index[c], special_token_2_index[d]

        score_ = scores[item_id].cpu().item()
        if 's' not in logists[user_id]:
            logists[user_id]['s'] = [(a, score_,)]
        else:
            logists[user_id]['s'].append((a, score_,))

        if a not in logists[user_id]:
            logists[user_id][a] = [(b, score_,)]
        else:
            logists[user_id][a].append((b, score_,))

        if b not in logists[user_id]:
            logists[user_id][b] = [(c, score_,)]
        else:
            logists[user_id][b].append((c, score_,))

        if c not in logists[user_id]:
            logists[user_id][c] = [(d, score_,)]
        else:
            logists[user_id][c].append((d, score_,))

import pickle
print('Saving graph-based logists ... ')
with open(data_dir + 'logists.pkl', 'wb') as f:
    pickle.dump(logists, f)
    f.close()
print('Finished!')

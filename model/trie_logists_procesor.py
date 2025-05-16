
import itertools

class Trie():
    def __init__(self):
        self.children = {}

class TrieMachine:
    def __init__(self, eos_token_id, sequences):
        self.eos_token_id = eos_token_id
        self.root = Trie()
        self.init_trie_with_sequences(sequences)

    def init_trie_with_sequences(self, sequences):
        for sequence in sequences:
            cur = self.root
            for token in sequence:
                if token not in cur.children:
                    cur.children[token] = Trie()
                cur = cur.children[token]
            cur.children[self.eos_token_id] = Trie()

    def get_root_node(self):
        return self.root


import torch
from transformers import LogitsProcessor


class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, tokenizer, num_beams, last_token=':'):
        self.trie = trie
        self.num_beams = num_beams
        self.last_token = tokenizer(last_token)['input_ids'][1:][0]
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids, scores):
        # Assuming mask is initialized outside or passed as an argument
        mask = torch.ones_like(scores, dtype=torch.bool)  # Example initialization, adjust as necessary

        for i, input_id in enumerate(input_ids):
            sequence = input_id.tolist()
            if self.last_token in sequence:
                index_from_end = len(sequence) - 1 - sequence[::-1].index(self.last_token)
                sub_sequence = sequence[index_from_end + 1:]
                cur = self.trie

                for s in sub_sequence:
                    if s in cur.children:
                        cur = cur.children[s]
                    else:
                        cur = None
                        break

                if cur:
                    next_states = cur.children
                else:
                    next_states = {self.eos_token_id}
                    scores[i, list(next_states)] = -1e10

                valid_token_ids = list(next_states)

                # Update mask
                mask[i, valid_token_ids] = False
                scores[i] = scores[i].masked_fill(mask[i], -float(1e12))

                if len(next_states) < self.num_beams:
                    complementary_ids = list(i for i in range(self.num_beams - len(next_states)))
                    scores[i, list(complementary_ids)] = -1e10

        return scores

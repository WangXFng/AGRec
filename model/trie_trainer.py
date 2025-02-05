from transformers import LlamaForCausalLM, Trainer


class TrieTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        user_ids = inputs.pop("user_ids", None)
        whole_word_ids = inputs.pop("whole_word_ids", None)
        outputs = model(**inputs, whole_word_ids=whole_word_ids, user_ids=user_ids)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
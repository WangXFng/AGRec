import torch
import pickle
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModelForCausalLM

_CONFIG_FOR_DOC = "LlamaConfig"


class AGRec(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = CrossEntropyLoss()
        self.temperature = 1.0
        self.alpha = 0.6
        self.finite_state = 0

    def set_hyper(self, temperature):
        self.temperature = temperature

    def setup_(self, tokenizer, new_tokens, alpha=None):
        self.tokenizer = tokenizer
        self.new_tokens = new_tokens
        if alpha is not None:
            self.alpha = alpha
        self.original_vocab_size = len(tokenizer) - len(new_tokens)

    def init_graph_embeddings(self, data_dir):
        with open(data_dir + 'logists.pkl', 'rb') as f:
            self.graph_logists = pickle.load(f)
            f.close()

    def decoding_graph_logists(self, input_ids, user_ids, finite_state):
        graph_logists = torch.zeros((input_ids.size(0), 1, len(self.new_tokens)), device=input_ids.device)

        for i, (input_id, user_id) in enumerate(zip(input_ids, user_ids)):
            user_cpu = str(user_id.cpu().item())
            # if user_cpu not in self.graph_logists: continue

            graph_logists[i][0] = 1e-5
            user_graph_logists = self.graph_logists[user_cpu]

            if finite_state == 0:
                indices_scores = user_graph_logists['s']
            else:
                ii = int(input_id.cpu().item()) - self.original_vocab_size # 128256
                if ii not in user_graph_logists:
                    continue
                indices_scores = user_graph_logists[ii]
            for (index, score) in indices_scores:
                graph_logists[i][0][index] = max(score, graph_logists[i][0][index])
        return graph_logists


    def ranking_loss(self, shift_logits, shift_labels):
        shift_logits = shift_logits.reshape(-1, self.config.vocab_size)
        shift_labels = shift_labels.reshape(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.loss_fct(shift_logits /self.temperature, shift_labels)
        return loss

    def total_loss(self, shift_logits, shift_labels):
        gen_loss = self.ranking_loss(shift_logits, shift_labels)
        loss = gen_loss
        return loss


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        user_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        if not self.training:
            if self.alpha != 0:
                # initial state s_0
                if len(input_ids[0]) > 3:
                    self.finite_state = 0

                # decoding FSMs into logits
                graph_logists = self.decoding_graph_logists(input_ids, user_ids, self.finite_state)
                cur_alpha = self.alpha

                logits[:,:,-len(self.new_tokens):] = graph_logists * cur_alpha + (1-cur_alpha) * logits[:,:,-len(self.new_tokens):]

                # state shift s_1, s_2, s_3
                self.finite_state += 1

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.total_loss(shift_logits, shift_labels)


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

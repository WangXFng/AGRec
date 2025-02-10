import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaConfig
from transformers import Trainer
from model.collator import Collator
from model.trie_trainer import TrieTrainer
import argparse
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training ,
)

import os
os.environ["WANDB_MODE"] = "disabled"
from model.modeling_trie import AGRec
from model.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()

    model = AGRec.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto")

    model.set_hyper(args.temperature)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0

    # --- load data ---
    train_data, valid_data = load_datasets(args)
    new_tokens = train_data.datasets[0].get_new_tokens()

    add_num = tokenizer.add_tokens(new_tokens)
    config = LlamaConfig.from_pretrained(args.base_model)
    config.vocab_size = len(tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except Exception as e:
        model.resize_token_embeddings(len(tokenizer))

    model.vocab_size = len(tokenizer)
    model.setup_(tokenizer, new_tokens, args.alpha)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainArgs = TrainingArguments(
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.per_device_batch_size,
        warmup_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=250,
        save_strategy="no",
        optim="adamw_hf",
        push_to_hub=False,
        save_total_limit=1,
        bf16=True,
        output_dir=args.output_dir,
        dataloader_pin_memory=False,
    )

    model.config.use_cache = False

    collator = Collator(args, tokenizer)
    trainer = TrieTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
        args=trainArgs,
        tokenizer=tokenizer,
    )

    model.print_trainable_parameters()
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

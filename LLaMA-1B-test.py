from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers import LlamaForCausalLM, LlamaTokenizer, LogitsProcessorList, LlamaConfig
from model.trie_logists_procesor import Trie, TrieMachine, TrieLogitsProcessor
from peft import PeftModel
from model.collator import TestCollator
from torch.utils.data import DataLoader
import argparse
from model.utils import *
from model.evaluate import get_metrics_results, get_topk_results_and_logs
from tqdm import tqdm
from model.prompt import all_prompt
from model.modeling_trie import AGRec
import time
# import optuna

def main(trial):

    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    model = AGRec.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto")

    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except Exception as e:
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


    model.print_trainable_parameters()

    test_data = load_test_dataset(args)
    all_items = test_data.get_all_items()
    new_tokens = test_data.get_new_tokens()

    model.setup_(tokenizer, new_tokens, args.alpha)
    if args.alpha != 0:
        model.init_graph_embeddings(args.data_path + '/' + args.dataset + '/')

    # # # ## ===================== Trie LogitsProcessor to avoid ghost items =====================
    encoded_sequences = []
    for sequence in all_items:
        token_ids = tokenizer.encode(sequence)
        encoded_sequences.append(token_ids[1:])
    trie = TrieMachine(tokenizer.eos_token_id, encoded_sequences).get_root_node()
    logits_processor = LogitsProcessorList([TrieLogitsProcessor(trie, tokenizer, args.num_beams, last_token=':')])

    collator = TestCollator(args, tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator
                             , num_workers=2, pin_memory=True)
    device = torch.device("cuda", 0)

    model.eval()
    metrics = args.metrics.split(",")
    all_prompt_results = []
    prompt_ids = range(len(all_prompt["seqrec"]))
    logs = []
    with torch.no_grad():
        for prompt_id in prompt_ids:
            metrics_results = {}
            total = 0
            test_loader.dataset.set_prompt(prompt_id)
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                while True:
                    try:
                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            user_ids=inputs["user_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            temperature=1,
                            num_beams=args.num_beams,
                            num_return_sequences=args.num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                            logits_processor = logits_processor,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        print("Out of memory!")
                        num_beams = num_beams - 1
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                all_device_topk_res, log = get_topk_results_and_logs(output, scores, targets, args.num_beams,
                                            all_items=all_items, user_ids=inputs["user_ids"])
                logs.extend(log)
                total += len(all_device_topk_res)

                batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % args.logging_step == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            with open(args.data_path + '/' + args.dataset + '/' + 'log_{}_{}.txt'.format(args.dataset,str(time.time())), 'w') as f:
                for log in logs:
                    f.write(str(log) + '\n')

                f.write('model.alpha: ' + str(model.alpha) + '\n')
                f.write(str(metrics_results) + '\n')
                f.close()

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    # if not os.path.exists(args.results_file): os.makedirs(args.results_file)
    # with open(args.results_file, "w") as f:
    #     json.dump(save_data, f, indent=4)
    # print("Save file: ", args.results_file)

    return max_results['ndcg@10']




if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    # study.optimize(main, n_trials=1)
    main(0)
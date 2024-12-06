import torch
from torch import nn
from torch.nn import functional as f
from datasets import Dataset, DatasetDict
import time
import tqdm
import numpy as np
import datetime
import random
from dataclasses import dataclass
from epoch_stats import EpochStats


PRINT_N_DIV = 50


def get_act_func(name, dim=None):
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "softmax":
        return nn.Softmax(dim=dim if dim is not None else -1)
    if name == "softplus":
        return nn.Softplus()
    if name == "none" or name is None:
        return None
    else:
        raise ValueError(f"unknown activation function '{name}'")


class Dataloader:
    def __init__(self, dataset, batch_size):
        self.ds = dataset
        self.bs = batch_size
        if isinstance(dataset, Dataset):
            self.schema = list(dataset.features.keys())
        if isinstance(dataset, dict):
            self.schema = list(dataset.keys())

    def _b_to_ten(self, b):
        ret = []
        for n in b.values():
            ret.append(torch.tensor(n))
        return ret

    def shuffle(self):
        self.ds = self.ds.shuffle()
        return self

    def __len__(self):
        return int(len(self.ds) / self.bs)

    def __iter__(self):
        batch = {}
        for step, n in enumerate(self.ds):
            for k, v in n.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)
            if (step + 1) % self.bs != 0:
                continue
            yield self._b_to_ten(batch)
            batch = {}


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class GenOutput:
    logits: torch.Tensor
    loss: torch.Tensor


def mask_tokens(
    input_ids, 
    vocab_size, 
    to_mask=0.15, 
    mask_token=4, 
    return_mask_only_decoder_ids=False, 
    chance_rand_token=0.2, 
    ignore_up_to=4
):
    """
    https://www.analyticsvidhya.com/blog/2022/09/fine-tuning-bert-with-masked-language-modeling/
    """
    inp_ids = []
    if return_mask_only_decoder_ids:
        dec_inp_ids = []
    lbs = []
    idx = 0
    for inp in np.array(input_ids):#.numpy():
        #actual_tokens = list(set(range(100)) - 
        #                    set(np.where((inp == 101) | (inp == 102) 
        #                        | (inp == 0))[0].tolist()))

        #actual_tokens = list(set(range(len(inp))) - 
        #                    set(np.where((inp <= ignore_up_to))[0].tolist()))
        actual_tokens = list(set(range(len(inp))))
                            #set(np.where((inp == 0))[0].tolist()))
        #We need to select 15% random tokens from the given list
        num_of_token_to_mask = int(len(actual_tokens) * to_mask)
        token_to_mask = np.random.choice(np.array(actual_tokens), 
                                        size=num_of_token_to_mask, 
                                        replace=False).tolist()
        #Now we have the indices where we need to mask the tokens
        if return_mask_only_decoder_ids:
            dec_inp = np.array([mask_token] * len(inp))
            dec_inp[token_to_mask] = inp[token_to_mask]
            dec_inp_ids.append(dec_inp)
        if random.random() < chance_rand_token:
            inp[token_to_mask] = random.randint(ignore_up_to+1, vocab_size)
        else:
            inp[token_to_mask] = mask_token
        inp_ids.append(inp)
        idx += 1
    if return_mask_only_decoder_ids:
        return inp_ids, dec_inp_ids
    return inp_ids


def _group_examples(examples, block_size, sparsify=True, pad_token=0, prefix_tokens=None, num_sparse_token=1):
    em_prefix_tokens = prefix_tokens is None
    if em_prefix_tokens:
        prefix_tokens = {"group_mask": []}
    concatenated_examples = {}
    for k, v in examples.items():
        cl = []
        group_mask = []
        for n in v:
        #    if len(n) < 1:
        #        continue
        #    if sparsify:
        #        cl += [doc_token] * num_sparse_token
        #        group_mask += [0] * num_sparse_token # TODO: change to 1
            cl += n
            group_mask += [1] * len(n)
            if sparsify:
                cl += [4] * num_sparse_token # token id chosen by trail and error, don't ask me why gpt2 needs a % token as the text seperator
                group_mask += [0] * num_sparse_token
        concatenated_examples[k] = cl
        #if "group_mask" not in concatenated_examples.keys():
        concatenated_examples["group_mask"] = group_mask
        #else:
        #    assert group_mask == concatenated_examples["group_mask"]
        if em_prefix_tokens:
            prefix_tokens[k] = []
    
    concatenated_examples["group_mask"] = group_mask
    num_keys = concatenated_examples.keys()
    print("num keys:", num_keys)

    prefix_len = 0
    for k, t in prefix_tokens.items():
        if prefix_len > 0:
            assert prefix_len == len(t), f"{prefix_len} != {len(t)}"
        prefix_len = len(t)
    block_size -= prefix_len

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [prefix_tokens[k] + t[i : i + block_size] for i in range(0, total_length, block_size)]
        #k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def preprocess_for_maskedlm(
    dataset, 
    tokenizer, 
    max_length, 
    num_proc=4, 
    remove_columns=None, 
    to_mask=.25, 
    text_field="text", 
    chance_rand_token=.2, 
    group_texts=True, 
    mask_token=None,
    pad_token=None, 
    sparsify=False, 
    prefix=None,
    switch_ii_decoder_ii=False,
):
    vocab_size = tokenizer.vocab_size
    if mask_token is None:
        mask_token = tokenizer.mask_token_id
    if pad_token is None:
        pad_token = tokenizer.pad_token_id
    print("MASK TOKEN", mask_token)
    def proc(examples):
        text = examples[text_field]
        if group_texts:
            encoding = tokenizer(text)
            #encoding = tokenizer([" ".join(x) for x in text])
        else:
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length) # try max_length=512
        
        if prefix is not None:
            if group_texts:
                prefix_tokens = tokenizer(prefix)
            else:
                raise NotImplementedError()
        else:
            prefix_tokens = None

        if group_texts:
            encoding = _group_examples(encoding, max_length, sparsify, pad_token=pad_token, prefix_tokens=prefix_tokens)

        inp_ids = encoding["input_ids"].copy()
        encoding["labels"] = encoding["input_ids"].copy()
        #encoding["decoder_input_ids"] = []
        #encoding["decoder_input_ids"].append(encoding["input_ids"].copy())
        
        tok_mask = mask_tokens(
            input_ids=inp_ids.copy(), # don't know if mask_tokens() mutates input_ids
            vocab_size=vocab_size, 
            to_mask=to_mask, 
            mask_token=mask_token, 
            return_mask_only_decoder_ids=False, 
            chance_rand_token=chance_rand_token,
            ignore_up_to=0
        )
        encoding["input_ids"] = tok_mask
        return encoding

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_key_masking(
    keys,
    dataset, 
    tokenizer, 
    max_length, 
    num_proc=4, 
    remove_columns=None, 
    to_mask=.15, 
    text_field="text",
    chance_rand_token=.2, 
    group_texts=True, 
    mask_token=None,
    pad_token=None, 
    sparsify=False, 
    prefix=None,
):
    mask_token = tokenizer.mask_token
    vocab_size = tokenizer.vocab_size
    #to_mask = .15
    #chance_rand_token = .2

    def proc(examples):
        out_text = examples[text_field]
        in_text = [
            " ".join([
                mask_token if tok in keys else tok
                for tok in text_i.split()
            ]) for text_i in out_text
        ]
        # in_text = out_text
        # to_mask_tokens = []
        # for n in tokenizer(out_text)["input_ids"]:
        #     to_mask_tokens += list(set(n))

        if group_texts:
            encoding = tokenizer(in_text)
            #encoding = tokenizer([" ".join(x) for x in text])
            encoding["labels"] = tokenizer(out_text)["input_ids"]
        else:
            encoding = tokenizer(in_text, padding="max_length", truncation=True, max_length=max_length) # try max_length=512
            encoding["labels"] = tokenizer(in_text, padding="max_length", truncation=True, max_length=max_length)
    


        if prefix is not None:
            if group_texts:
                prefix_tokens = tokenizer(prefix)
            else:
                raise NotImplementedError()
        else:
            prefix_tokens = None

        if group_texts:
            encoding = _group_examples(encoding, max_length, sparsify, pad_token=pad_token, prefix_tokens=prefix_tokens)

        tok_mask = mask_tokens(
            input_ids=encoding["input_ids"].copy(),
            vocab_size=vocab_size, 
            to_mask=to_mask, 
            mask_token=mask_token, 
            return_mask_only_decoder_ids=False, 
            chance_rand_token=chance_rand_token,
            ignore_up_to=mask_token
        )
        encoding["input_ids"] = tok_mask

        return encoding

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_with_given_labels_train_test_wrap(
    dataset, 
    tokenizer, 
    labels, 
    label2id, 
    max_length, 
    one_label_only, 
    num_proc=4, 
    remove_columns=None, 
    text_field="text", 
    default_teacher_forcing=True,
    #teacher_forcing_prefix="The correct label is "
    teacher_forcing_prefix="",
    doc_pad_tokens=False,
    prefix=None,
    postfix=None,
    incontext_dict: dict=None,
    move_incontext_to_decoder=False,
):
    train = preprocess_with_given_labels(
        dataset["train"], 
        tokenizer, 
        labels, 
        label2id, 
        max_length, 
        one_label_only, 
        num_proc, 
        remove_columns, 
        text_field, 
        default_teacher_forcing,
        #teacher_forcing_prefix="The correct label is "
        teacher_forcing_prefix,
        doc_pad_tokens,
        prefix,
        postfix,
        True,
        incontext_dict,
        move_incontext_to_decoder
    )
    test = preprocess_with_given_labels(
        dataset["test"], 
        tokenizer, 
        labels, 
        label2id, 
        max_length, 
        one_label_only, 
        num_proc, 
        remove_columns, 
        text_field, 
        default_teacher_forcing,
        #teacher_forcing_prefix="The correct label is "
        teacher_forcing_prefix,
        doc_pad_tokens,
        prefix,
        postfix,
        False,
        incontext_dict,
        move_incontext_to_decoder
    )
    encoded_dataset = DatasetDict({
        "train": train,
        "test": test
    })
    return encoded_dataset


def preprocess_with_given_labels(
    dataset, 
    tokenizer, 
    labels, 
    label2id, 
    max_length, 
    one_label_only, 
    num_proc=4, 
    remove_columns=None, 
    text_field="text", 
    default_teacher_forcing=True,
    #teacher_forcing_prefix="The correct label is "
    teacher_forcing_prefix="",
    doc_pad_tokens=False,
    prefix=None,
    postfix=None,
    train=True,
    incontext_dict: dict=None,
    move_incontext_to_decoder=False
):
    SEPERATOR_TOKEN = "[SEP]"

    if prefix is None:
        prefix = ""
    if postfix is None:
        postfix = ""
    def proc(examples):
        text = examples[text_field]
        if incontext_dict is not None:
            new_text, used_context_keys = [], []
            for text_i in text:
                x_context = []
                for tok in text_i.split():
                    if tok in incontext_dict:
                        #print("NEW CONTEXT INSERTED")
                        used_context_keys.append(tok)
                        if incontext_dict[tok] is not None:
                            if not move_incontext_to_decoder:
                                x_context.append(f"{tok} is defined as {incontext_dict[tok]}")
                            else:
                                x_context.append(incontext_dict[tok])
                if not move_incontext_to_decoder:
                    new_text.append(text_i + SEPERATOR_TOKEN.join(x_context))
                else:
                    new_text.append(x_context)
            if not move_incontext_to_decoder:
                text = new_text
           
        if doc_pad_tokens:
            encoding = tokenizer(text)
            encoding = _doc_pad(encoding, max_length)
        else:
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        if one_label_only:
            encoding["labels"] = [label2id[n] for n in examples["label"]]
        else:
            labels_batch = {k: v for k, v in examples.items() if k in labels}
            labels_matrix = np.zeros((len(text), len(labels)))
            for idx, label in enumerate(labels):
                labels_matrix[:, idx] = labels_batch[label]
            encoding["labels"] = labels_matrix.tolist()

        num_pos_examples = 5
        num_neg_examples = 5
        
        if move_incontext_to_decoder and incontext_dict is not None:
            neg_examples_keys = [n for n in incontext_dict.keys() if n not in used_context_keys]
            neg_examples = []
            for _ in range(num_neg_examples):
                val = None
                while val is None:
                    key = random.choice(neg_examples_keys)
                    val = incontext_dict[key]
                neg_examples.append(val)
            new_text = random.choices(new_text, k=num_pos_examples)
            pos_exp_stack = []
            for n_t in new_text:
                t = tokenizer(n_t, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
                pos_exp_stack.append(t)
            neg_exp_stack = []
            for n_t in neg_examples:
                t = tokenizer(n_t, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
                neg_exp_stack.append(t)
            stack_t = torch.Tensor([pos_exp_stack, neg_exp_stack])#.permute(2, 0, 1)
            encoding["decoder_input_ids"] = stack_t.tolist()
        else:
            if default_teacher_forcing:
                encoding["decoder_input_ids"] = tokenizer([teacher_forcing_prefix + str(n) for n in encoding["labels"]], padding="max_length", truncation=True, max_length=max_length)["input_ids"]
            else:
                encoding["decoder_input_ids"] = encoding["input_ids"].copy()
            
        return encoding

    return dataset.map(proc, batched=True, num_proc=1, remove_columns=remove_columns)


STRING_BATCH_INDEX = 0

def _get_batch_item(batch, batch_schema, item, device):
    if item not in batch_schema:
        return None
    if STRING_BATCH_INDEX:
        idx = item
    else:
        idx = batch_schema.index(item)
    return batch[idx].to(device)
    

def _get_model_args(batch, batch_schema, items, device):
    ret = {}
    for n in items:
        e = _get_batch_item(batch, batch_schema, n, device)
        if e is not None:
            ret[n] = e
    return ret


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def dump_model(model, checkpoint_path: str, i: int):
    iter_dir = f"{checkpoint_path}/iter_{i:02}/"
    model.save_pretrained(iter_dir, safe_serialization=False)
    print(f"  model dumped at location '{iter_dir}'")


def train_epoch(
    epoch_idx,
    model,
    device,
    dataloader,
    id2label,
    forward_args,
    optimizer=None,
    scheduler=None,
    eval=False,
    backprop=True,
    empty_cache=False,
    calc_metrics=True,
    use_tqdm=False
):
    stats = EpochStats("train", id2label, per_class_f1=True)
    t0 = time.time()
    if eval:
        model.eval()
    else:
        model.train()
    batch_schema = dataloader.schema
    last_elapse = 0
    print_n = len(dataloader) // PRINT_N_DIV
    _iter_fn = enumerate(dataloader)
    if use_tqdm:
        _iter_fn = tqdm.tqdm(_iter_fn, total=len(dataloader))
    for step, batch in _iter_fn:
        if not use_tqdm and not eval and (step % print_n == 0) and not step == 0:
            c_time = time.time()
            elapsed = c_time - t0
            last_step_time = int(round(elapsed - last_elapse))
            remaining = (len(dataloader) - step) / print_n * last_step_time
            last_elapse = elapsed
            print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:>8}, Remaining: {:>8}.".format(step, 
                                                                                            len(dataloader), 
                                                                                            format_time(elapsed), 
                                                                                            format_time(remaining)))

        model.zero_grad()
        model_args = _get_model_args(batch, batch_schema, forward_args, device)
        labels = _get_batch_item(batch, batch_schema, "labels", device)
        
        outputs = model(**model_args)
        logits = outputs.logits
        loss = outputs.loss

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        stats.add_score("loss", loss.item())
        if calc_metrics:
            stats.flat_metrics(logits, labels)

        if backprop:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if empty_cache:
            torch.cuda.empty_cache()

    print(f"\n  Average {'evaluation' if eval else'training'} scores:")
    for k, v in stats.calc_avg_scores().items():
        print(f"    {k}: {v}")
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    return stats


def train_run(
    model,
    device,
    train_dataloader,
    test_dataloader,
    id2label,
    forward_args,
    optimizer,
    scheduler,
    num_epochs=10,
    empty_cache=False,
    calc_metrics=True,
    use_tqdm=False,
    checkpoint_path=None
):
    train_stats, test_stats = [], []
    for epoch_i in range(1, num_epochs+1):
        print("\n======== Epoch {:} / {:} ========".format(epoch_i, num_epochs))

        ts = train_epoch(
            epoch_i,
            model,
            device,
            train_dataloader,
            id2label,
            forward_args,
            optimizer,
            scheduler,
            empty_cache=empty_cache,
            calc_metrics=calc_metrics,
            use_tqdm=use_tqdm
        )
        train_stats.append(ts)

        es = train_epoch(
            epoch_i,
            model,
            device,
            test_dataloader,
            id2label,
            forward_args,
            eval=True,
            backprop=False,
            empty_cache=empty_cache,
            calc_metrics=calc_metrics,
            use_tqdm=use_tqdm
        )
        test_stats.append(es)

        if checkpoint_path is not None:
            dump_model(model, checkpoint_path, epoch_i)
            
    print("\nTraining completed!")
    return train_stats, test_stats

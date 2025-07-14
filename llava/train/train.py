import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Literal
from tqdm import tqdm
import torch
import ujson
import transformers
import tokenizers
import numpy as np

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_anyres_image, tokenizer_image_token

from PIL import Image
from llava.utils import get_logger, compute_metrics_special_token
logger= get_logger(__name__)

local_rank = None

TEMPLATE_DICT = {
    "template_0": "",
    "template_1": "Consider this paragraph: ",
    "template_2": "Consider this paragraph to answer: ",
    "template_3": "This may contain useful information to help answer the question correctly: ",
    "template_4": "This may contain relevant information to help answer the question correctly:",
    "template_rank" : {"pre": "Considering this paragraph: ", "post": " assess if it is [Relevant] or [No Relevant] to the question and then give a short answer."}
}

def is_debug():
    return int(os.environ.get('DEBUG', 0))

# use this function instead the standard print, to avoid verbose output in the logs
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def set_random_seed(seed, deterministic=False, benchmark=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default=None)
    siglip: bool = field(default=False)
    wikillava: bool = field(default=False)


@dataclass
class DataArguments:
    data_path_train: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path_eval: str = field(default=None, metadata={"help": "Path to the eval data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    mask_paragraph: bool = field(default=False, metadata={"help": "in visual rag mask the loss for the retrieved paragraph"})
    mask_answer: bool = field(default=False, metadata={"help": "loss masking for the answer"}) # TODO: not implemented yet
    resampler: Optional[Literal['multiply', 'divide', 'vanilla']] = field(default='vanilla', metadata={"help": "Resampler type"})
    # seed: Optional[int] = field(default=42, metadata={"help": "set seeds for reproducibility"})
    prefix_relevant: Optional[Literal['val_multiturn_relevant_hard', 'val_multiturn_relevant_soft']] = field(default='relevant_hard', metadata={"help": "Name of the prefix dataset to use in the evaluation"})
    template_paragraph: Optional[Literal['template_0', 'template_1', 'template_2', 'template_3', 'template_4', 'template_rank']] = field(default='template_0', metadata={"help": "Name of the prefix before the paragraph"})
    generator_training: bool = True
    llava_instruct_multiplier: int = 1

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    llm_backbone: str = field(default='vicuna')
    llm_pad_token: str = field(default=None)
    visual_rag: bool = True
    standard_loss: bool = True # define if you want use the original loss of the model
    weight_loss_special: int = 5 # weight of the first loss component
    special_token_retrieval: int = 128251 # special token for retrieval
    special_token_no_retrieval: int = 128250 # special token for no retrieval
    special_token_relevant: int = 128253 # special token for retrieval
    special_token_no_relevant: int = 128252 # special token for no retrieval
    compute_metrics_special_tokens: Optional[Literal['retrieval', 'relevant']] = field(default=None, metadata={"help": "special token type to compute metrics"})

def find_sequence(lst, seq):
    lst_len = len(lst)
    seq_len = len(seq)

    # Iterate over the list with a sliding window
    for i in range(lst_len - seq_len + 1):
        # Check if the slice matches the sequence
        if lst[i:i+seq_len] == seq:
            return i, i+seq_len

    return 0, 0

def create_visual_rag_mask(mask_paragraph, mask_answer, target, special_tokens, tokenized_template= None, instruct_template=None):
    if mask_paragraph or mask_answer:
        # check if present the paragraph (start and end), then mask that part
        start_paragraph_token = special_tokens['start_paragraph']
        end_paragraph_token = special_tokens['end_paragraph']
        # sanity check on the paragraph structure
        if torch.sum(target == start_paragraph_token).item() == torch.sum(target == end_paragraph_token).item():
            number_paragraph= torch.sum(target == start_paragraph_token).item()
        else:
            raise ValueError("The paragraph structure is not correct, the number of start and end tokens should be the same.")
        
        # mask the paragraph
        # TODO: check the code when you are using more than 1 paragraph
        for p_index_ith, p in enumerate(range(number_paragraph)):
            if start_paragraph_token in target:
                start_idx = torch.nonzero(torch.eq(target, start_paragraph_token))[p_index_ith]
                end_idx = torch.nonzero(torch.eq(target, end_paragraph_token))[p_index_ith]
                target[start_idx[p]:end_idx[p]+1] = IGNORE_INDEX
                start_idx_template, end_idx_template= find_sequence(target.tolist(), tokenized_template)
                target[start_idx_template:end_idx_template] = IGNORE_INDEX
                if instruct_template is not None:
                    start_idx_template, end_idx_template= find_sequence(target.tolist(), instruct_template)
                    target[start_idx_template:end_idx_template] = IGNORE_INDEX

    if mask_answer:
        # mask from 2 to a special token both excluded (Retrieval) or (No Retrieval)
        number_paragraph= torch.sum(target == start_paragraph_token).item()
        list_start= []
        list_end= []
        start_excluded_ret_token = special_tokens['ret_token_answer']
        start_excluded_no_ret_token = special_tokens['no_ret_token_answer']
        end_excluded_tokens = 2

        for answer_idx_ith, el in enumerate(target):
            if el in [start_excluded_ret_token, start_excluded_no_ret_token]:
                list_start.append(answer_idx_ith)
            if el == end_excluded_tokens:
                list_end.append(answer_idx_ith)
        if len(list_start) != len(list_end):
            raise ValueError("The number of start and end tokens should be the same.")

        for st,nd in zip(list_start, list_end):
            target[st+1:nd] = IGNORE_INDEX
    
    return target


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def efficient_tokenizer_expansion(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    ):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    rank0_print(f"Number of new special tokens: {num_new_tokens}")
    model.resize_token_embeddings(len(tokenizer)) # , pad_to_multiple_of=64

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg.clone()
        output_embeddings[-num_new_tokens:] = output_embeddings_avg.clone()
        input_embeddings = input_embeddings.contiguous()
        output_embeddings = output_embeddings.contiguous()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3(
        sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets

    sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
    #sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds= [rounds[0]] + [rounds[idx] + rounds[idx+1] for idx in range(1, len(rounds)-1, 2)]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(tokenizer(rou, add_special_tokens=False).input_ids)

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3_1(
    data_args: DataArguments,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Mask targets
    sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
    #sep = conv.sep + conv.roles[1] + ": "
    visual_rag_special_tokens = {
        'start_paragraph': 128254,
        'end_paragraph': 128255,
        'ret_token_answer': 128251,
        'no_ret_token_answer': 128250,
    }
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds= [rounds[0]] + [rounds[idx] + rounds[idx+1] for idx in range(1, len(rounds)-1, 2)]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            
            tokenized_template= []
            if 'rank' in data_args.template_paragraph:
                tokenized_template_pre= tokenizer.encode(TEMPLATE_DICT[data_args.template_paragraph]['pre'])[1:]
                tokenized_template_post= tokenizer.encode(TEMPLATE_DICT[data_args.template_paragraph]['post'])[1:]
                target = create_visual_rag_mask(data_args.mask_paragraph, data_args.mask_answer, target, visual_rag_special_tokens, 
                                                tokenized_template_pre, tokenized_template_post)
            else:
                if data_args.template_paragraph != 'template_0' and 'rank' not in data_args.template_paragraph:
                    tokenized_template= tokenizer.encode(TEMPLATE_DICT[data_args.template_paragraph])[1:]
                target = create_visual_rag_mask(data_args.mask_paragraph, data_args.mask_answer, target, visual_rag_special_tokens, 
                                                tokenized_template)

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(tokenizer(rou, add_special_tokens=False).input_ids)

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i > 0: round_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len= cur_len + len(tokenizer(sep, add_special_tokens=False).input_ids)

        # if cur_len > tokenizer.model_max_length: print(f"WARNING: max length context")
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    data_args: DataArguments,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    visual_rag_special_tokens = {
        'start_paragraph': 31998,
        'end_paragraph': 31999,
        'ret_token_answer': 31994,
        'no_ret_token_answer': 31995,
    }
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            
            target = create_visual_rag_mask(data_args.mask_paragraph, data_args.mask_answer, target, visual_rag_special_tokens)

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2 

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN # we not used also 'what is in the photo'
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    # bos and image are IGNORE_INDEX
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    data_args: DataArguments,
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1:
        return preprocess_llama_3_1(data_args, sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(data_args, sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, split, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: ModelArguments):
        super(LazySupervisedDataset, self).__init__()
        self.split = split
        
        for file in os.listdir(data_path):
            if split in file:
                with open(os.path.join(data_path, file), 'r') as f:
                    list_data_dict = ujson.load(f)
        # list_data_dict = json.load(open(data_path, "r"))
        
        # correct the image path
        all_list_data_dict = []
        count_data_format_problem = 0
        for sample in tqdm(list_data_dict, mininterval=1, total=len(list_data_dict)):
            try:
                sample = fill_abs_path_train(sample)
                all_list_data_dict.append(sample)
            except:
                count_data_format_problem = count_data_format_problem + 1
        rank0_print(f"count data format problem {count_data_format_problem}")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = all_list_data_dict
        self.data_args = data_args
        self.model_args = model_args
        self.siglip= model_args.siglip
        
        if self.split == 'val':
            random.seed(42)
            random.shuffle(self.list_data_dict)
            self.list_data_dict = self.list_data_dict[:2500]

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if self.model_args.wikillava:
                image = Image.open(image_file).convert('RGB')
            else:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, self.siglip) # torch.Size([5, 3, 336, 336])
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


            # the two different preprocessors produce the same results
            # if self.siglip:
            #     image= processor(images=image, return_tensors="pt")['pixel_values'][0]
            # else:
            # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            self.data_args,
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if self.siglip:
                crop_size= self.data_args.image_processor.size
            else:
                crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
                
        if 'dataset_name' in instances[0]:
            batch['dataset_name'] = [instance['dataset_name'] for instance in instances]
        
        if 'original_question' in instances[0]:
            batch['original_question'] = [instance['original_question'] for instance in instances]
        
        if 'original_image' in instances[0]:
            batch['original_image'] = [instance['original_image'] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(split='train', tokenizer=tokenizer,
                                data_path=data_args.data_path_train,
                                data_args=data_args, model_args=model_args)
    
    eval_dataset = LazySupervisedDataset(split='val', tokenizer=tokenizer,
                            data_path=data_args.data_path_eval,
                            data_args=data_args, model_args=model_args)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def fill_abs_path_special_tokens(sample):
    # TODO: insert here the path to the dataset images
    cineca = os.environ.get('CINECA', False)
    prefix_data = ''
    prefix_data_gqa = ''
    
    
    if 'gqa' in sample['image']:
        sample['image'] = prefix_data_gqa + sample['image'].split('/')[-1]
        
    if 'encyclopedic' in sample['image'] or 'infoseek' in sample['image']:
        sample['image'] = os.path.join(prefix_data, 'visualRAG')

    if not sample['image'].startswith('/'):
        sample['image'] = prefix_data + sample['image']

    return sample

def fill_abs_path_train(sample):
    # TODO: insert here the path to the dataset images
    cineca = os.environ.get('CINECA', False)
    prefix_data = ''
    
    if 'encyclopedic' in sample['image'] or 'infoseek' in sample['image']:
        sample['image'] = os.path.join(prefix_data, 'visualRAG')
    if 'train2014' in sample['image']:
        sample['image'] = os.path.join(prefix_data, 'coco', sample['image'])
    elif 'train2017' in sample:
        sample['image'] = os.path.join(prefix_data, sample['image'])
    if not sample['image'].startswith('/'):
        sample['image'] = os.path.join(prefix_data, sample['image'])

    return sample


class Custom_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: ModelArguments):
        super(Custom_LazySupervisedDataset, self).__init__()
        self.data = {}
        for file in os.listdir(data_path):
            if file.startswith(split):
                with open(os.path.join(data_path, file), 'r') as f:
                    if 'okvqa' in file:
                        title = 'okvqa'
                    elif 'infoseek' in file:
                        title = 'infoseek'
                    elif 'mix' in file:
                        title = 'mix'
                    elif 'encyclopedic' in file:
                        title = 'encyclopedic'
                    self.data[title] = ujson.load(f)
        
        self.list_data_dict = []
        self.list_data_dict_ret = []
        self.list_data_dict_no_ret = []
        
        variable_template= TEMPLATE_DICT[data_args.template_paragraph]
        rank0_print(f'Paragraph template activated... {variable_template}')

        if 'okvqa' in self.data:
            dataset = self.data['okvqa']
            rank0_print('Loading OKVQA dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['image']
                cleaned_sample = fill_abs_path_train(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['id']
                cleaned_sample['ret_token'] = not 'No' in sample['need_retrieval']    
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_no_ret.append(cleaned_sample)
            rank0_print('OKVQA dataset loaded...')
        
        if 'infoseek' in self.data:
            dataset = self.data['infoseek']
            rank0_print('Loading Infoseek dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                if data_args.generator_training:
                    cleaned_sample['image'] = sample['image']
                else:
                    cleaned_sample['image'] = sample['image_mo_srv_path']
                    cleaned_sample = fill_abs_path_train(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                if 'rank' in data_args.template_paragraph:
                    cleaned_sample['conversations'][2]['value'] = variable_template['pre'] + cleaned_sample['conversations'][2]['value'] + variable_template['post']
                else:
                    cleaned_sample['conversations'][2]['value'] = variable_template + cleaned_sample['conversations'][2]['value']
                cleaned_sample['id'] = sample['__key__'] if data_args.generator_training else sample['id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']                
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_ret.append(cleaned_sample)
            rank0_print('Infoseek dataset loaded...')

        if 'encyclopedic' in self.data:
            dataset = self.data['encyclopedic']
            rank0_print('Loading EVQA dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                if data_args.generator_training:
                    cleaned_sample['image'] = sample['image']
                else:
                    cleaned_sample['image'] = sample['image_path']
                    cleaned_sample = fill_abs_path_train(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                if 'rank' in data_args.template_paragraph:
                    cleaned_sample['conversations'][2]['value'] = variable_template['pre'] + cleaned_sample['conversations'][2]['value'] + variable_template['post']
                else:
                    cleaned_sample['conversations'][2]['value'] = variable_template + cleaned_sample['conversations'][2]['value']
                cleaned_sample['id'] = sample['unique_id'] if data_args.generator_training else sample['id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                      
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_ret.append(cleaned_sample)
            rank0_print('EVQA dataset loaded...')
        
        if 'mix' in self.data:
            dataset = self.data['mix']
            dataset = dataset * data_args.llava_instruct_multiplier
            rank0_print('Loading LLaVA-Instruct dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                try:
                    cleaned_sample['image'] = sample['image']
                    cleaned_sample = fill_abs_path_train(cleaned_sample)
                except:
                    cleaned_sample['image'] = Image.new('RGB', (224, 224))
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_no_ret.append(cleaned_sample)
            rank0_print('LLaVA-Instruct dataset loaded...')

        rank0_print("Formatting inputs...Skip in lazy mode")

        # resampler policy
        rank0_print(f"Resampling strategy: {data_args.resampler}")
        
        if data_args.resampler != 'vanilla':
            # multiply ret to have the same dimension of no_ret
            if data_args.resampler == 'multiply':
                self.list_data_dict_ret = self.list_data_dict_ret * int(len(self.list_data_dict_no_ret) / len(self.list_data_dict_ret))
            # divide no_ret to have the same dimension of ret
            elif data_args.resampler == 'divide':
                random.shuffle(self.list_data_dict_no_ret)
                self.list_data_dict_no_ret = self.list_data_dict_no_ret[:len(self.list_data_dict_ret)]

        # concat ret to no_ret & shuffle
        self.list_data_dict = self.list_data_dict_no_ret + self.list_data_dict_ret
        random.shuffle(self.list_data_dict)
        
        if is_debug():
            rank0_print(f"Dataset dimension is reducted because we are in debug mode.")
            self.list_data_dict = self.list_data_dict[:200]
        rank0_print(f"Dataset dimension: {len(self.list_data_dict)}")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.siglip= model_args.siglip

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            if not isinstance(image_file, Image.Image):
                image = Image.open(image_file).convert('RGB')
            else:
                image = image_file
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, self.siglip) # torch.Size([5, 3, 336, 336])
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


            # the two different preprocessors produce the same results
            # if self.siglip:
            #     image= processor(images=image, return_tensors="pt")['pixel_values'][0]
            # else:
            #     image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            self.data_args,
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if self.siglip:
                crop_size= self.data_args.image_processor.size
            else:
                crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict
    
class Custom_LazySupervisedDatasetSpecialVal(Dataset):
    def __init__(self, data_path: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: ModelArguments):
        super(Custom_LazySupervisedDatasetSpecialVal, self).__init__()
        self.data = {}
        for file in os.listdir(data_path):
            if file.startswith(split):
                with open(os.path.join(data_path, file), 'r') as f:
                    if 'infoseek' in file:
                        title = 'infoseek'
                    elif 'encyclopedic' in file:
                        title = 'encyclopedic'
                    elif 'gqa' in file:
                        title = 'gqa'
                    self.data[title] = ujson.load(f)
                    
        self.list_data_dict = []
        variable_template= TEMPLATE_DICT[data_args.template_paragraph]
        rank0_print(f'Paragraph template activated in val... {variable_template}')
        
        if 'gqa' in self.data:
            dataset = self.data['gqa']
            rank0_print('Loading GQA dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['query_img_path'][0]
                cleaned_sample = fill_abs_path_special_tokens(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['imageId']
                cleaned_sample['dataset_name'] = sample['dataset']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']

                self.list_data_dict.append(cleaned_sample)
            rank0_print('GQA Special Val dataset loaded...')

        if 'infoseek' in self.data:
            dataset = self.data['infoseek']
            rank0_print('Loading Infoseek dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['query_img_path'][0]
                cleaned_sample = fill_abs_path_special_tokens(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                if split != 'special_tokens': 
                    cleaned_sample['conversations'][2]['value'] = variable_template + cleaned_sample['conversations'][2]['value']
                cleaned_sample['id'] = sample['data_id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                cleaned_sample['dataset_name'] = sample['dataset']
                
                self.list_data_dict.append(cleaned_sample)
            rank0_print('Infoseek Special Val dataset loaded...')

        if 'encyclopedic' in self.data:
            dataset = self.data['encyclopedic']
            rank0_print('Loading EVQA dataset...')
            for idx_enc, sample in enumerate(tqdm(dataset, mininterval=1, total=len(dataset))):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['query_img_path'][0]
                cleaned_sample = fill_abs_path_special_tokens(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                if split != 'special_tokens':
                    cleaned_sample['conversations'][2]['value'] = variable_template + cleaned_sample['conversations'][2]['value']
                cleaned_sample['id'] = 'encyclopedic_' + str(idx_enc)
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                cleaned_sample['dataset_name'] = 'encyclopedic'
                
                self.list_data_dict.append(cleaned_sample)
            rank0_print('EVQA Special Val dataset loaded...')
        
        if is_debug():
            rank0_print('DEBUG MODE: only a subset of the validation is performed.')
            self.list_data_dict= self.list_data_dict[:50]

        rank0_print(f"Dataset Special Val dimension: {len(self.list_data_dict)}")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.siglip= model_args.siglip

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        original_question = sources['conversations'][0]['value'].split('<image>\n')[1]
        original_image = sources['image']
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        dataset_name= sources[0]['dataset_name']
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            if not isinstance(image_file, Image.Image):
                image = Image.open(image_file).convert('RGB')
            else:
                image = image_file
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, self.siglip) # torch.Size([5, 3, 336, 336])
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


            # the two different preprocessors produce the same results
            # if self.siglip:
            #     image= processor(images=image, return_tensors="pt")['pixel_values'][0]
            # else:
            #     image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            self.data_args,
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if self.siglip:
                crop_size= self.data_args.image_processor.size
            else:
                crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        data_dict['dataset_name'] = dataset_name
        data_dict['original_question'] = original_question
        data_dict['original_image'] = original_image
        return data_dict
    
    
def custom_make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning.""" 
    train_dataset = Custom_LazySupervisedDataset(split='train_multiturn',
                                                 tokenizer=tokenizer,
                                                 data_path=data_args.data_path_train,
                                                 data_args=data_args, model_args=model_args)
    
    if data_args.compute_metrics_special_tokens == 'retrieval':
        rank0_print(f'Custom_LazySupervisedDatasetSpecialVal: is not ready for template with rank')
        eval_dataset = Custom_LazySupervisedDatasetSpecialVal(split='special_tokens',
                                                    tokenizer=tokenizer,
                                                    data_path=data_args.data_path,
                                                    data_args=data_args, model_args=model_args)
    elif data_args.compute_metrics_special_tokens == 'relevant':
        rank0_print(f'Custom_LazySupervisedDatasetSpecialVal: is not ready for template with rank')
        eval_dataset = Custom_LazySupervisedDatasetSpecialVal(split=data_args.prefix_relevant,
                                            tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            data_args=data_args, model_args=model_args)
    elif data_args.generator_training:
        eval_dataset = Custom_LazySupervisedDataset(split='val_multiturn',
                                                tokenizer=tokenizer,
                                                data_path=data_args.data_path_eval,
                                                data_args=data_args, model_args=model_args)
    else:
        eval_dataset = None
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    set_random_seed(training_args.seed)
    rank0_print(f"Original loss: {training_args.standard_loss}")

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    # use the last option
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # for all the version of llama 3 not expand the dictionary with unk token
        # it can create problem in stage two when importing the configuration value of the vocab size
        if "llama_3" not in training_args.llm_backbone:
            if tokenizer.unk_token is None:
                rank0_print("resize embedding dimesion")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(unk_token="[UNK]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        # select the correct PAD token for llama_3_1 and llama_3
        if training_args.llm_backbone == "llama_3_1":
            rank0_print(f"pad token: {training_args.llm_pad_token}")
            if training_args.llm_pad_token == 'end_of_text':
                tokenizer.pad_token_id= 128001
            elif training_args.llm_pad_token == 'eot':
                tokenizer.pad_token_id= 128009
            elif training_args.llm_pad_token == 'pad':
                tokenizer.pad_token_id= 128004
            else:
                raise ValueError(f"Unknown llm_pad_token")
                        
        elif training_args.llm_backbone == "llama_3":
            if training_args.llm_pad_token == 'eos':
                tokenizer.pad_token = tokenizer.eos_token
            elif training_args.llm_pad_token == 'pad':
                tokenizer.pad_token_id= 128003
            else:
                tokenizer.pad_token = tokenizer.unk_token

        else:
            tokenizer.pad_token = tokenizer.unk_token
        
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # count parameters in the model
    count_par_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count_par = sum(p.numel() for p in model.parameters())
    rank0_print(f"Trainable parameters: {count_par_trainable}")
    rank0_print(f"Total parameters: {count_par}")
    
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids]
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        if model_args.s2:
            rank0_print("Using S2")
            model.config.s2 = model_args.s2
            model.config.s2_scales = model_args.s2_scales
        if model_args.siglip:
            rank0_print("Using SigLIP")
            model.config.siglip = model_args.siglip
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if training_args.visual_rag:
        if not 'template_paragraph' in model.config.to_dict():
            model.config.template_paragraph = data_args.template_paragraph
        else:
            data_args.template_paragraph = model.config.template_paragraph
            
        rank0_print("Using Visual RAG dataset")
        training_args.model_name_or_path= model_args.model_name_or_path
        data_args.compute_metrics_special_tokens = training_args.compute_metrics_special_tokens
        data_module = custom_make_supervised_data_module(tokenizer=tokenizer,
                                                data_args=data_args, model_args=model_args)
    else:
        rank0_print("Using original LLaVA dataset")
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                data_args=data_args, model_args=model_args)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    compute_metrics=compute_metrics_special_token if training_args.compute_metrics_special_tokens else None,
                    **data_module)

    if training_args.compute_metrics_special_tokens == 'retrieval':
        metrics = trainer.evaluate_special_token()
        print(f"\nCheckpoint -- [{model_args.model_name_or_path}] -- Tokens [Ret]")
        print(f'Accuracy global: {metrics["eval_accuracy_global"]}')
        print(f'Accuracy TOP2: {metrics["eval_accuracy_top2"]}')
        print(f'Accuracy GQA: {metrics["eval_accuracy_gqa"]}')
        print(f'Accuracy INFOSEEK: {metrics["eval_accuracy_infoseek"]}')
        print(f'Accuracy ENCYCLOPEDIC: {metrics["eval_accuracy_encyclopedic"]}')
        return
    elif training_args.compute_metrics_special_tokens == 'relevant':
        metrics = trainer.evaluate_special_token()
        print(f"\nCheckpoint -- [{model_args.model_name_or_path}] -- Tokens [Rel]")
        print(f"\nDataset -- [{data_args.prefix_relevant}] -- Tokens [Rel]")
        print(f'Accuracy ENCYCLOPEDIC: {metrics["eval_accuracy_encyclopedic"]}')
        print(f'Accuracy TOP2: {metrics["eval_accuracy_top2"]}')
        print(f'Accuracy relevant: {metrics["eval_accuracy_relevant"]}')
        print(f'Accuracy no relevant: {metrics["eval_accuracy_no_relevant"]}')
        return 
        
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
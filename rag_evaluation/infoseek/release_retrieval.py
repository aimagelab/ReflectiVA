import argparse
from braceexpand import braceexpand
import torch
import os
import ujson
from tqdm import tqdm
from torch.utils.data import DataLoader
import webdataset as wds
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, collate_and_pad_input_ids
from torch.utils.data import DataLoader
import gc
import random
from spacy.lang.en import English
import faiss
import numpy as np
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor

IMAGE_TOKEN = f"{DEFAULT_IMAGE_TOKEN}\n\n"
CONTEXT_PROMPT = "# {context}\n"
RET_TOKEN = 128251
REL_TOKEN = 128253
TEMPLATE = "Consider this paragraph: "

global CURRENT_ID
CURRENT_ID = 0

class Retriever:
    def __init__(self, args):
        self.args = args
        print("Loading retrieval index")
        self.index = faiss.read_index(os.path.join(args.index_path, 'knn.index'))
        self.values = ujson.load(open(os.path.join(args.index_path_json, 'knn.json'), 'r'))
        print("Done loading retrieval index")

    def retrieve(self, query, k):
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.astype(np.float32)
        _, indexes = self.index.search(query, k=100)
        chosen_k = indexes[0, :k].tolist()
        return [self.values[k][0] for k in chosen_k]
    
    
def uniform_passages_of_sentences(paragraphs, n=100):
    spacy_model = English()
    spacy_model.add_pipe("sentencizer")
    text = paragraphs

    sentences = spacy_model(text).sents

    passages = []
    passage = []
    tokens_in_passage = 0
    for sent in sentences:
        if tokens_in_passage + len(sent) > n:
            if len(passage) > 0:
                passages.append(' '.join(passage))
                passage = [sent.text]
                tokens_in_passage = len(sent)
            else:
                passages.append(sent.text)
        else:
            passage.append(sent.text)
            tokens_in_passage += len(sent)

    if len(passage) > 0:
        passages.append(' '.join(passage))

    return passages


def concat_paragraph(paragraphs_list, question, sections, conv_mode, config, tokenizer):
    template = TEMPLATE
    qs = IMAGE_TOKEN + question
    conv_relevant = conv_templates[conv_mode].copy()
    conv_relevant.append_message(conv_relevant.roles[0], qs)
    qs = '[Retrieval]'
    conv_relevant.append_message(conv_relevant.roles[1], qs)
    qs = template + '<paragraph>'
    for idx in paragraphs_list:
        qs += sections[idx]
    qs +='</paragraph>'
    
    if args.short_prompt:
        qs += ". Give a short answer."
        
    conv_relevant.append_message(conv_relevant.roles[0], qs)
    conv_relevant.append_message(conv_relevant.roles[1], None)
    prompt = conv_relevant.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[1:].unsqueeze(0).to(device='cuda', non_blocking=True)


def inference(args, model, input_ids, images, image_sizes):
    output_ids = model.generate(
        input_ids,
        images=images.to(dtype=torch.float16, device='cuda', non_blocking=True),
        image_sizes=image_sizes,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True)
    
    return output_ids


def create_data_loader(
    shard_list, 
    tokenizer, 
    image_processor, 
    model_config, 
    args
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retriever = Retriever(args)
    clip_name_or_path = args.retriever_path
    
    if args.use_clip_to_retrieve:
        clip = AutoModel.from_pretrained(clip_name_or_path, torch_dtype=torch.float16).to(device).eval()
        clip_preprocessor = AutoImageProcessor.from_pretrained(clip_name_or_path)
    elif args.use_eva_to_retrieve:
        clip = AutoModel.from_pretrained(clip_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
        clip_preprocessor = CLIPImageProcessor.from_pretrained(args.retriever_processor_path)
    
    with open(args.kb_wikipedia_path, 'r') as f:
        wikipedia = ujson.load(f)
    
    def process_sample(sample):
        global CURRENT_ID
        if CURRENT_ID >= args.start_idx and CURRENT_ID < args.end_idx:
            template = TEMPLATE
            
            question = sample.pop('question').decode('utf-8')
            with torch.no_grad():
                if args.use_clip_to_retrieve:
                    pixel_values = torch.from_numpy(clip_preprocessor(sample['img.jpg']).pixel_values[0])[None].to(dtype=torch.float16, device=clip.device)
                    image_features = clip.get_image_features(pixel_values)
                else:
                    pixel_values = torch.from_numpy(clip_preprocessor(sample['img.jpg']).pixel_values[0])[None].to(dtype=torch.float16, device=clip.device)
                    image_features = clip.encode_image(pixel_values)
                
                image_features /= image_features.norm(dim=-1, p=2)
                entity_urls = retriever.retrieve(image_features, k=args.entity_k)
            
            sections = []
            for entity_url in entity_urls:
                sections.extend(uniform_passages_of_sentences(wikipedia[entity_url]['wikipedia_content'], n=300))

            total_qs_relevant = []
            for section in sections:
                qs_retrieval = IMAGE_TOKEN + question
                conv_relevant = conv_templates[args.conv_mode].copy()
                qs_relevant = IMAGE_TOKEN + question
                conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
                qs_relevant = '[Retrieval]'
                conv_relevant.append_message(conv_relevant.roles[1], qs_relevant)
                qs_relevant = template + '<paragraph>'
                qs_relevant += section + '</paragraph>'
                    
                conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
                conv_relevant.append_message(conv_relevant.roles[1], None)
                prompt_relevant = conv_relevant.get_prompt()
                
                input_ids_relevant = tokenizer_image_token(prompt_relevant, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                total_qs_relevant.append(input_ids_relevant[1:])

            conv_retrieval = conv_templates[args.conv_mode].copy()
            conv_retrieval.append_message(conv_retrieval.roles[0], qs_retrieval)
            conv_retrieval.append_message(conv_retrieval.roles[1], None)
            prompt_retrieval = conv_retrieval.get_prompt()

            input_ids_retrieval = tokenizer_image_token(prompt_retrieval, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            
            pil_img = sample.pop('img.jpg')
            image_tensor = process_images([pil_img], image_processor, model_config)[0]

            out = dict(
                img=image_tensor,
                img_size=pil_img.size,
                retrieval_input_ids=input_ids_retrieval[1:],
                relevant_input_ids=total_qs_relevant,
                question=question,
                sections=sections
            )
            
        else:
            out = None
            CURRENT_ID += 1

        return out


    def collate_fn(batch, tokenizer=tokenizer):
        out = {k: [example[k] for example in batch] for k in list(batch[0].keys())}
        out['retrieval_input_ids'] = collate_and_pad_input_ids(out['retrieval_input_ids'], tokenizer.pad_token_id, 'left')
        out['relevant_input_ids'] = [collate_and_pad_input_ids(el.unsqueeze(0), tokenizer.pad_token_id, 'left') for el in out['relevant_input_ids'][0]]
        return out
    
    dataset = wds.DataPipeline(
            wds.SimpleShardList(shard_list),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.split_by_node,
            wds.split_by_worker,
            wds.decode("pil", handler=wds.warn_and_continue),
            wds.map(process_sample, handler=wds.warn_and_continue),
        )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False
    )
    return data_loader


def eval_model(args):
    global CURRENT_ID
    
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name if args.model_name else get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    model.cuda()

    part = int(os.environ.get('PART', None))
    total_part = int(os.environ.get('TOTAL_PART', None)) + 1
    print(f'Computing split {part} of the dataset...')
    slicing = 73620 // total_part
    if (part+1) == total_part:
        args.start_idx = slicing * part
        args.end_idx = 73620
    else:
        args.start_idx = slicing * part
        args.end_idx = slicing * part + slicing
    print(f'Processing Element from {args.start_idx} to {args.end_idx}...')
        
    shard_list = braceexpand(args.shard_path)
    data_loader = create_data_loader(shard_list, tokenizer, image_processor, model.config, args)
            
    out_data = []
    for sample in tqdm(data_loader, mininterval=1):
        question_raw = sample['question']
        retrieval_ids = sample['retrieval_input_ids'].to(device='cuda', non_blocking=True)
        relevant_ids_list = sample['relevant_input_ids']
        images = sample['img'][0].unsqueeze(0)
        image_sizes = sample['img_size'][0]
        sections = sample['sections'][0]

        # Retrieval Forward [FIRST STAGE]
        with torch.inference_mode():
            output_ids = inference(args, model, retrieval_ids, images, image_sizes)

        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        torch.cuda.empty_cache()
        gc.collect()
        
        relevant_paragraphs_detected = []
        if RET_TOKEN == output_ids[0][0].item() or args.fix_ret_token:
            for id_context, element in enumerate(relevant_ids_list):
                relevant_ids = element.to(device='cuda', non_blocking=True)
            
                # Relevant Forward [SECOND STAGE]
                with torch.inference_mode():
                    output_ids = inference(args, model, relevant_ids, images, image_sizes)

                if REL_TOKEN == output_ids[0][0].item():
                    relevant_paragraphs_detected.append(id_context)
            
            if len(relevant_paragraphs_detected) == 0:
                relevant_paragraphs_detected = [ random.randint(0,len(sections)-1) ]

            with torch.inference_mode():
                concat_relevant_paraghraphs = concat_paragraph(relevant_paragraphs_detected, question_raw[0], sections, args.conv_mode, model.config, tokenizer)
                output_ids = inference(args, model, concat_relevant_paraghraphs, images, image_sizes)
        
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
        del output_ids
        torch.cuda.empty_cache()
        gc.collect()

        for i in range(len(answers)):
            sample = {"data_id": sample['__key__'][i], "prediction": answers[i].strip()}
            out_data.append(sample)
        
        CURRENT_ID += 1
    
    with open(args.answers_file, "w") as f:
        f.write(ujson.dumps(out_data))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # ReflectiVA hyperparameters
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="llama_3_1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # ReflectiVA evaluation parameteres
    parser.add_argument("--entity_k", type=int, default=1)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--index_path_json", type=str)
    parser.add_argument('--short_prompt', action='store_true')
    parser.add_argument("--kb_wikipedia_path", type=str)
    parser.add_argument('--use_clip_to_retrieve', action='store_true')
    parser.add_argument('--use_eva_to_retrieve', action='store_true')
    parser.add_argument("--retriever_path", type=str)
    parser.add_argument("--retriever_processor_path", type=str)
    parser.add_argument("--shard_path", type=str)
    args = parser.parse_args()
    
    if args.use_eva_to_retrieve:
        retriever_model = 'eva'
    elif args.use_clip_to_retrieve:
        retriever_model = 'clip'
        
    index = 'Image2Text_Summary'
    
    args.answers_file = f'output/Reflectiva_infoseek_{retriever_model}_index_{index}'
    part = os.environ.get('PART', None)
    os.makedirs(args.answers_file, exist_ok=True)
    args.answers_file = f'{args.answers_file}/split_{part}.json'
    eval_model(args)
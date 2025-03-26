import argparse
import torch
import os
import ujson
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, collate_and_pad_input_ids
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
from tqdm import tqdm
from functools import partial
import gc
import random
import faiss
import numpy as np
from spacy.lang.en import English

IMAGE_TOKEN = f"{DEFAULT_IMAGE_TOKEN}\n\n"
RET_TOKEN = 128251
REL_TOKEN = 128253
TEMPLATE = "Consider this paragraph: "


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

class Retriever:
    def __init__(self, args):
        self.args = args
        print("Loading retrieval index")
        self.index = faiss.read_index(
            os.path.join(args.index_path, 'knn.index'))
        self.values = ujson.load(
            open(os.path.join(args.index_path_json, 'knn.json'), 'r'))
        print("Done loading retrieval index")

    def retrieve(self, query, k):
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.astype(np.float32)
        _, indexes = self.index.search(query, k=100)
        chosen_k = indexes[0, :k].tolist()
        return [self.values[k][0] for k in chosen_k]

class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor, model_config):
        self.args = args
        self.data_path = args.data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = args.conv_mode
        self.template = TEMPLATE
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.retriever = Retriever(args)
        clip_name_or_path = args.retriever_path
        
        if args.use_clip_to_retrieve:
            self.clip = AutoModel.from_pretrained(clip_name_or_path, torch_dtype=torch.float16).to(device).eval()
            self.clip_preprocessor = AutoImageProcessor.from_pretrained(clip_name_or_path)
        elif args.use_eva_to_retrieve:
            self.clip = AutoModel.from_pretrained(clip_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
            self.clip_preprocessor = CLIPImageProcessor.from_pretrained(args.retriever_processor_path)
        
        with open(args.kb_wikipedia_path, 'r') as f:
            self.wikipedia = ujson.load(f)
            
        print(f'Loading from {self.data_path}...')
        with open(self.data_path, 'r') as f:
            self.all_samples = ujson.load(f)
        print(f'Loading completed...')
        
        self.entities = self.all_samples
        part = int(os.environ.get('PART', None))
        total_part = int(os.environ.get('TOTAL_PART', None)) + 1
        print(f'Computing split {part} of the dataset...')
        slicing = len(self.all_samples) // total_part
        if (part+1) == total_part:
            self.all_samples = self.all_samples[slicing * part:]
        else:
            self.all_samples = self.all_samples[slicing * part:slicing * part + slicing]

    def __getitem__(self, index):
        sample = self.all_samples[index]
        image = Image.open(sample["related_images"]).convert('RGB')
        sample['dataset_image_ids'] = sample['dataset_image_ids']
        question = sample['question']

        total_qs_relevant = []
        with torch.no_grad():           
            if args.use_clip_to_retrieve:
                pixel_values = torch.from_numpy(self.clip_preprocessor(image).pixel_values[0])[None].to(dtype=torch.float16, device=self.clip.device)
                image_features = self.clip.get_image_features(pixel_values)
            else:
                pixel_values = torch.from_numpy(self.clip_preprocessor(image).pixel_values[0])[None].to(dtype=torch.float16, device=self.clip.device)
                image_features = self.clip.encode_image(pixel_values)

            image_features /= image_features.norm(dim=-1, p=2)
            entity_urls = self.retriever.retrieve(image_features, k=args.entity_k)

            sections = []
            for entity_url in entity_urls:
                for section in self.wikipedia[entity_url]['section_texts']:
                    if len(section.strip().lower()) < 5:
                        continue
                    sections.append(section)
                   
            for section in sections:
                section = section.strip()
                qs_retrieval = IMAGE_TOKEN + question
                conv_relevant = conv_templates[args.conv_mode].copy()
                qs_relevant = IMAGE_TOKEN + question
                conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
                qs_relevant = '[Retrieval]'
                conv_relevant.append_message(conv_relevant.roles[1], qs_relevant)
                if 'rank' in self.model_config.template_paragraph:
                    qs_relevant = self.template['pre'] + '<paragraph>'
                    qs_relevant += section + '</paragraph>' + self.template['post']
                else:
                    qs_relevant = self.template + '<paragraph>'
                    qs_relevant += section + '</paragraph>'
                    
                conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
                conv_relevant.append_message(conv_relevant.roles[1], None)
                prompt_relevant = conv_relevant.get_prompt()
                
                input_ids_relevant = tokenizer_image_token(prompt_relevant, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                total_qs_relevant.append(input_ids_relevant[1:])

        conv_retrieval = conv_templates[args.conv_mode].copy()
        conv_retrieval.append_message(conv_retrieval.roles[0], qs_retrieval)
        conv_retrieval.append_message(conv_retrieval.roles[1], None)
        prompt_retrieval = conv_retrieval.get_prompt()

        input_ids_retrieval = tokenizer_image_token(prompt_retrieval, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        sample['image_size'] = image.size
        sample['image_tensor'] = process_images([image], self.image_processor, self.model_config)[0]
        sample['retrieval_input_ids'] = input_ids_retrieval[1:]
        sample['relevant_input_ids'] = total_qs_relevant
        sample['sections'] = sections

        return sample

    def __len__(self):
        return len(self.all_samples)


def create_data_loader(args, tokenizer, image_processor, model_config):
    dataset = CustomDataset(args, tokenizer, image_processor, model_config)

    def collate_fn(tokenizer, batch):
        out = {k: [example[k] for example in batch]
               for k in list(batch[0].keys())}
        out['retrieval_input_ids'] = collate_and_pad_input_ids(out['retrieval_input_ids'], tokenizer.pad_token_id, 'left')
        out['relevant_input_ids'] = [collate_and_pad_input_ids(el.unsqueeze(0), tokenizer.pad_token_id, 'left') for el in out['relevant_input_ids'][0]]
        out['image_tensor'] = torch.stack(out['image_tensor'], dim=0)
        return out

    collate_fn = partial(collate_fn, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name if args.model_name else get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    model.cuda()

    data_loader = create_data_loader(args, tokenizer, image_processor, model.config)

    out_data = []
    for batch in tqdm(data_loader, mininterval=1, total=len(data_loader)):
        question_raw = batch['question']
        retrieval_ids = batch['retrieval_input_ids'].to(device='cuda', non_blocking=True)
        relevant_ids_list = batch['relevant_input_ids']
        images = batch['image_tensor']
        image_sizes = batch['image_size']
        reference = batch['answer']
        question_type = batch['question_type']
        sections = batch['sections'][0]

        # Retrieval Forward [FIRST STAGE]
        with torch.inference_mode():
            output_ids = inference(args, model, retrieval_ids, images, image_sizes)

        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        torch.cuda.empty_cache()
        gc.collect()
        
        relevant_paragraphs_detected = []
        if 128251 == output_ids[0][0].item() or args.fix_ret_token:
            for id_context, element in enumerate(relevant_ids_list):
                relevant_ids = element.to(device='cuda', non_blocking=True)
            
                # Relevant Forward [SECOND STAGE]
                with torch.inference_mode():
                    output_ids = inference(args, model, relevant_ids, images, image_sizes)

                if 128253 == output_ids[0][0].item():
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
            sample = {"question": question_raw[i],
                      "reference": reference[i].split('|'),
                      "answers": answers[i],
                      "question_type": question_type[i]
                      }
            out_data.append(sample)

    with open(args.answers_file, "w") as f:
        f.write(ujson.dumps(out_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # ReflectiVA hyperparameters
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
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
    args = parser.parse_args()
    
    if args.use_eva_to_retrieve:
        retriever_model = 'eva'
    elif args.use_clip_to_retrieve:
        retriever_model = 'clip'
        
    index = 'I2I'

    args.answers_file = f'output/Reflectiva_evqa_{retriever_model}_index_{index}'
    part = os.environ.get('PART', None)
    os.makedirs(args.answers_file, exist_ok=True)
    args.answers_file = f'{args.answers_file}/split_{part}.json'
    eval_model(args)
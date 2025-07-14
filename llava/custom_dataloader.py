from torch.utils.data import Dataset, DataLoader
import os
import ujson
from tqdm import tqdm
from PIL import Image


def fill_abs_path(sample):
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

class VisualRagDataset(Dataset):
    def __init__(self, data_path, split = 'train'):
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

        if 'okvqa' in self.data:
            dataset = self.data['okvqa']
            print('Loading OKVQA dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['image']
                cleaned_sample = fill_abs_path(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['id']
                cleaned_sample['ret_token'] = not 'No' in sample['need_retrieval']
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_no_ret.append(cleaned_sample)
            print('OKVQA dataset loaded...')
        
        if 'infoseek' in self.data:
            dataset = self.data['infoseek']
            print('Loading Infoseek dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['image_mo_srv_path']
                cleaned_sample = fill_abs_path(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_ret.append(cleaned_sample)
            print('Infoseek dataset loaded...')

        if 'encyclopedic' in self.data:
            dataset = self.data['encyclopedic']
            print('Loading EVQA dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                cleaned_sample['image'] = sample['image_path']
                cleaned_sample = fill_abs_path(cleaned_sample)
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_ret.append(cleaned_sample)
            print('EVQA dataset loaded...')
        
        if 'mix' in self.data:
            dataset = self.data['mix']
            print('Loading LLaVA-Instruct dataset...')
            for sample in tqdm(dataset, mininterval=1, total=len(dataset)):
                cleaned_sample = {}
                try:
                    cleaned_sample['image'] = sample['image']
                    cleaned_sample = fill_abs_path(cleaned_sample)
                except:
                    cleaned_sample['image'] = Image.new('RGB', (224, 224))
                cleaned_sample['conversations'] = sample['conversations']
                cleaned_sample['id'] = sample['id']
                cleaned_sample['ret_token'] = '[Retrieval]' in sample['conversations'][1]['value']
                self.list_data_dict.append(cleaned_sample)
                self.list_data_dict_no_ret.append(cleaned_sample)
            print('LLaVA-Instruct dataset loaded...')

        print("Formatting inputs...Skip in lazy mode")

        # check in all dataloader
        count = 0
        not_found = []
        for el in tqdm(self.list_data_dict, mininterval=1, total=len(self.list_data_dict)):
            if not isinstance(el['image'], Image.Image) and not os.path.exists(el['image']):
                count += 1
                not_found.append(el['image'])
        print(count)
        print(not_found)

    def __len__(self):
        return len(self.cleaned_data)

    def __getitem__(self, idx):
        id = self.cleaned_data[idx]['id']
        query_img = self.cleaned_data[idx]['image']
        conversation = self.cleaned_data[idx]['conversations']
        return id, query_img, conversation
    
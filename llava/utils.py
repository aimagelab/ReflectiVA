import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import ujson
import requests

from llava.constants import LOGDIR
import torch
import torch.distributed as dist

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

def compute_metrics_special_token(all_inputs, dataset_name, custom_metric, compute_result = True, original_question=None, original_image=None):
    """
    Consider the logits and labels. Using the second one to find the position of the special retrieval tokens.
    Detect if the top one associated logits is the same as the label. If so, it is a correct match and improve the accuracy.
    Detect also if the ret or not ret are in the top 2 positions. 

    Args:
        all_inputs
        tokenizer
        dataset_name
        special_token_ids

    Returns:
        mertrics: dictionary of metrics
    """
    dataset_name= dataset_decode_names(dataset_name)
    # dataset_name= [el for el in dataset_name[0]]
    assert all_inputs.label_ids.shape[0] == all_inputs.predictions.shape[0]
    custom_metric.update(all_inputs.predictions, all_inputs.label_ids, dataset_name, original_question=original_question, original_image=original_image)
    
    if compute_result:
        return custom_metric.compute()
    

class SpecialAccuracyMetric:
    def __init__(self, special_tokens, file_to_save_wrong_predictions=None):
        self.count_gqa = []
        self.count_infoseek = []
        self.count_encyclopedic = []
        self.count_global = []
        
        self.count_relevant = []
        self.count_no_relevant = []
        self.acc_relevant = []
        self.acc_no_relevant = []
        
        self.acc_gqa = []
        self.acc_infoseek = []
        self.acc_encyclopedic = []
        self.acc_global = []
        self.acc_top2 = []
        if file_to_save_wrong_predictions is not None:
            self.file_to_save_wrong_predictions = file_to_save_wrong_predictions
            self.wrong_predictions = []
        else:
            self.file_to_save_wrong_predictions = None
            self.wrong_predictions = None
        
        self.special_tokens = special_tokens


    def update(self, preds, label_ids, dataset_name, original_question=None, original_image=None):
        count_gqa = 0
        count_infoseek = 0
        count_encyclopedic = 0
        count_global = 0
        
        acc_gqa = 0
        acc_infoseek = 0
        acc_encyclopedic = 0
        acc_global = 0
        acc_top2 = 0
        
        count_relevant = 0
        count_no_relevant = 0
        acc_relevant = 0
        acc_no_relevant = 0
        
        for i in range(label_ids.shape[0]):
            for j in range(label_ids.shape[1]):
                if label_ids[i][j] in self.special_tokens:
                    count_global+=1
                    gt= label_ids[i][j]
                    pred= preds[i][j-1].argmax()
                
                    if pred in self.special_tokens:
                        acc_top2+=1
                    if gt == pred:
                        acc_global+=1
                        if dataset_name[i] == "gqa":
                            acc_gqa+=1
                        elif dataset_name[i] == "infoseek":
                            acc_infoseek+=1
                        elif dataset_name[i] == "encyclopedic":
                            acc_encyclopedic+=1
                            
                        if pred == self.special_tokens[0]:
                            acc_relevant += 1
                        elif pred == self.special_tokens[1]:
                            acc_no_relevant += 1
                            
                    if dataset_name[i] == "gqa":
                        count_gqa+=1
                    elif dataset_name[i] == "infoseek":
                        count_infoseek+=1
                    elif dataset_name[i] == "encyclopedic":
                        count_encyclopedic+=1
                        if self.file_to_save_wrong_predictions is not None:
                            error = {'question': original_question[i], 'image': original_image[i], 'gt': gt.item(), 'prediction': pred.item()}
                            self.wrong_predictions.append(error)
                        
                    if label_ids[i][j] == self.special_tokens[0]:
                        count_relevant+=1
                    if label_ids[i][j] == self.special_tokens[1]:
                        count_no_relevant+=1
                        
        
        self.acc_gqa.append(acc_gqa)
        self.acc_infoseek.append(acc_infoseek)
        self.acc_encyclopedic.append(acc_encyclopedic)
        self.acc_global.append(acc_global)
        self.acc_top2.append(acc_top2)
        
        self.count_gqa.append(count_gqa)
        self.count_infoseek.append(count_infoseek)
        self.count_encyclopedic.append(count_encyclopedic)
        self.count_global.append(count_global)
        
        self.count_relevant.append(count_relevant)
        self.count_no_relevant.append(count_no_relevant)
        self.acc_relevant.append(acc_relevant)
        self.acc_no_relevant.append(acc_no_relevant)

    def compute(self):
        metrics={}
        metrics["accuracy_global"] = sum(self.acc_global)/sum(self.count_global)*100
        metrics["accuracy_top2"] = sum(self.acc_top2)/sum(self.count_global)*100    

        if sum(self.count_gqa) > 0:
            metrics["accuracy_gqa"] = sum(self.acc_gqa)/sum(self.count_gqa)*100
        else:
            metrics["accuracy_gqa"] = 0.0
        if sum(self.count_infoseek) > 0:
            metrics["accuracy_infoseek"] = sum(self.acc_infoseek)/sum(self.count_infoseek)*100
        else:
            metrics["accuracy_infoseek"] = 0.0
        if sum(self.count_encyclopedic) > 0:
            metrics["accuracy_encyclopedic"] = sum(self.acc_encyclopedic)/sum(self.count_encyclopedic)*100
        else:
            metrics["accuracy_encyclopedic"] = 0.0
        
        if self.file_to_save_wrong_predictions:
            with open(self.file_to_save_wrong_predictions, 'w') as f:
                ujson.dump(self.wrong_predictions, f)
            
        if sum(self.count_relevant) > 0:
            metrics["accuracy_relevant"] = sum(self.acc_relevant)/sum(self.count_relevant)*100
        else:
            metrics["accuracy_relevant"] = 0.0
        if sum(self.count_no_relevant) > 0:
            metrics["accuracy_no_relevant"] = sum(self.acc_no_relevant)/sum(self.count_no_relevant)*100
        else:
            metrics["accuracy_no_relevant"] = 0.0
        
        return metrics

def dataset_encode_names(dataset_name):
    output_list= []
    for el in dataset_name:
        if el == 'gqa':
            output_list.append(1)
        elif el == 'infoseek':
            output_list.append(2)
        elif el == 'encyclopedic':
            output_list.append(3)
        else:
            raise ValueError("Not correct encoding for dataset names")

    dataset_name_output= torch.tensor(output_list)
    return dataset_name_output

def dataset_decode_names(dataset_name):
    output_list= []
    for el in dataset_name:
        if el == 1:
            output_list.append('gqa')
        elif el == 2:
            output_list.append('infoseek')
        elif el == 3:
            output_list.append('encyclopedic')
        else:
            raise ValueError("Not correct decoding for dataset names")
    return output_list

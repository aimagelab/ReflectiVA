"""
InfoSeek Evaluation Script.
Credits: https://github.com/edchengg/infoseek_eval/blob/main/infoseek_eval.py
"""

import os
import re
import json
import ujson
import string
from typing import Any, Dict, Generator, List, Tuple, Union
import argparse
from tqdm import tqdm
from os.path import isfile, join
from os import listdir
import os


def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction exactly matches the normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(
    metric_fn,
    prediction: str,
    ground_truths: List[str]
) -> Union[int, bool]:
    """Compute the maximum score of a prediction over a list of ground truths using a given metric function."""
    return max(
        metric_fn(prediction, ground_truth) for ground_truth in ground_truths
    )


def in_range(number: float, range_list: Tuple[float, float]) -> bool:
    """Check if a number is within the specified range (inclusive)."""
    min_num, max_num = range_list
    return min_num <= number <= max_num


def safe_division(x: float, y: float) -> float:
    """Divide x by y, returning 0 if y is 0."""
    return x / y if y != 0 else 0


def metric_numerical_range(
    pred: Union[float, Tuple[float, float], List[float]],
    answer: Union[float, Tuple[float, float], List[float]],
    tolerance: float = 0.1,
) -> int:
    """Scores numerical questions based on ranges and tolerances.

    1) First, convert single number answer to a range with +/- tolerance.
    2) If prediction is a single number, return 1 if it's in the answer range, 0
    otherwise.
    3) If prediction is a range, return 1 if the range is in the answer range or
    if the IOU
        (overlap between prediction and answer range) > 0.5, 0 otherwise.

    Args:
        pred: A list/tuple of 2 numbers or a single number.
        answer: A list/tuple of 2 numbers or a single number.
        tolerance: A float value for the tolerance range (default: 0.1).

    Returns:
        int: 1 if conditions are met, 0 otherwise.
    """
    answer = list(answer) if isinstance(answer, tuple) else answer
    pred = list(pred) if isinstance(pred, tuple) else pred

    if not isinstance(answer, list):
        answer = [answer * (1 - tolerance), answer * (1 + tolerance)]

    # Prediction is a single number
    if not isinstance(pred, list):
        return 1 if in_range(pred, answer) else 0

    # Prediction is a range
    if answer[0] <= pred[0] <= answer[1] and answer[0] <= pred[1] <= answer[1]:
        return 1
    else:
        iou = range_intersection_over_union(pred, answer)
        return 1 if iou >= 0.5 - 1e-12 else 0


def process_numerical_answer(string_number: str) -> Union[float, List[float]]:
    """Parses numerical answer string into numbers (a single number or a range).

    1) Clean the string and extract numbers;
    2) if there are 2 numbers, return a range as [minimum value, maximum value]
        else if there is 1 number, return a single number
        else return [0, 0]

    Args:
        string_number: A string representing a numerical answer.

    Returns:
        A single digit or a list with 2 numbers.
    """
    # Clean string
    string_number = clean_str_range(string_number)
    numerical_numbers_tmp = re.findall(
        r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string_number
    )
    numerical_numbers_tmp = [
        n.replace(',', '').strip('.') for n in numerical_numbers_tmp
    ]
    numerical_numbers = []
    for n in numerical_numbers_tmp:
        if n.count('.') > 1:
            n = n.split('.')[0]
            numerical_numbers.append(float(n))
        else:
            numerical_numbers.append(float(n))

    # Use the first 2 numbers
    if len(numerical_numbers) > 2:
        numerical_numbers = numerical_numbers[:2]

    if len(numerical_numbers) == 2:
        first_val = numerical_numbers[0]
        second_val = numerical_numbers[1]
        return [first_val, second_val] if first_val <= second_val else first_val
    elif len(numerical_numbers) == 1:
        return numerical_numbers[0]
    else:
        return [0, 0]


def find_all(s: str, c: str) -> Generator[int, None, None]:
    """Find all occurrences of a character in a string and return their indices.

    Args:
        s: The input string to search.
        c: The character to search for.

    Yields:
        int: The index of the next occurrence of the character.
    """
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)


def clean_str_range(text: str) -> str:
    """Clean range expression in a string (e.g., '9-10' --> '9 - 10').

    Args:
        text: The input string containing the range expression.

    Returns:
        str: The cleaned string with proper spacing around the hyphen.
    """
    idx_list = list(find_all(text, '-'))
    idx_replace = [
        idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()
    ]
    new_str = ''.join(
        ' - ' if idx in idx_replace else s for idx, s in enumerate(text)
    )
    return new_str


def range_intersection_over_union(
    x_list: List[float], y_list: List[float]
) -> float:
    """Calculate the intersection over union (IOU) of two ranges."""
    min_1, max_1 = min(x_list), max(x_list)
    min_2, max_2 = min(y_list), max(y_list)

    overlap = max(0.0, min(max_1, max_2) - max(min_1, min_2))
    length_x = (max_1 - min_1) + 1e-12
    length_y = (max_2 - min_2) + 1e-12
    iou = safe_division(overlap, length_x + length_y - overlap)
    return iou


def evaluate_quantity(
    quantity_pred: List[Union[float, List[float]]],
    quantity_answer: List[List[float]],
) -> List[int]:
    """Evaluate numerical predictions against numerical answers."""
    return [
        metric_numerical_range(pred, ans)
        for pred, ans in zip(quantity_pred, quantity_answer)
    ]


def evaluate_entity(
    entity_pred: List[str], entity_answer: List[List[str]]
) -> List[int]:
    """Evaluate entity predictions against entity answers.

    Criteria: Maximum score of exact match to entity answer.

    Args:
        entity_pred: prediction of a string
        entity_answer: a list of string answer reference

    Returns:
        List: 0 or 1
    """
    return [
        metric_max_over_ground_truths(exact_match_score, pred, ans)
        for pred, ans in zip(entity_pred, entity_answer)
    ]


def evaluate_time(
    time_pred: List[str], time_answer: List[List[str]]
) -> List[int]:
    """Evaluate time predictions against time answers.

    Criteria:
    1) +/- one year --> correct
    2) if asking for date, but the year is correct --> correct

    Args:
        time_pred: prediction of time
        time_answer: a list of time reference

    Returns:
        List: 0 or 1
    """
    return [
        metric_max_over_ground_truths(exact_match_score, pred, ans)
        for pred, ans in zip(time_pred, time_answer)
    ]


def evaluation(
    predictions: List[Dict[str, Any]], qid2example: Dict[str, Dict[str, Any]]
) -> Tuple[List[int], List[int], List[int]]:
    """Evaluate predictions against ground truth answers.

    Separate questions into time, numerical, and string categories.

    Args:
        predictions: A list of predictions.
        qid2example: A mapping from question ID to ground truth examples.

    Returns:
        Tuple[List[int], List[int], List[int]]: Lists of scores for time,
        quantity, and entity predictions.
    """
    time_pred, quantity_pred, entity_pred = [], [], []
    time_answer, quantity_answer, entity_answer = [], [], []

    for p in predictions:
        quid = p['data_id']
        if quid not in qid2example:
            continue
        example = qid2example[quid]
        pred = p['prediction']
        answer = example['answer_eval']
        question_type = example['question_type'].lower()
        if question_type == 'time':
            time_pred.append(pred)
            time_answer.append(answer)
        elif question_type == 'numerical':
            pred_range = process_numerical_answer(pred)
            answer_range = [float(a) for a in answer]
            quantity_pred.append(pred_range)
            quantity_answer.append(answer_range)
        else:
            entity_pred.append(pred)
            entity_answer.append(answer)

    score_time = evaluate_time(time_pred, time_answer)
    score_quantity = evaluate_quantity(quantity_pred, quantity_answer)
    score_entity = evaluate_entity(entity_pred, entity_answer)
    return score_time, score_quantity, score_entity


def get_results(
    predictions: List[Dict[str, Any]], qid2example: Dict[str, Dict[str, Any]]
) -> Tuple[float, float, float, float]:
    """Get evaluation scores for predictions.

    Args:
        predictions: A list of predictions.
        qid2example: A mapping from question ID to ground truth examples.

    Returns:
        Tuple[float, float, float, float]: Final scores for time, quantity,
        entity, and overall predictions.
    """
    score_time, score_quantity, score_entity = evaluation(
        predictions, qid2example
    )
    final_score_time = safe_division(sum(score_time), len(score_time))
    final_score_quantity = safe_division(
        sum(score_quantity), len(score_quantity))
    final_score_entity = safe_division(sum(score_entity), len(score_entity))
    final_score = safe_division(
        sum(score_time + score_quantity + score_entity),
        len(score_time + score_quantity + score_entity),
    )
    return final_score, final_score_time, final_score_quantity, final_score_entity


def harmonic_mean(*args: float) -> float:
    """Calculate the harmonic mean of the input arguments."""
    args_safe = [a if a != 0 else 1e-12 for a in args]
    hmean = len(args_safe) / sum((1.0 / val) for val in args_safe)
    return hmean


def evaluate_infoseek(
    predictions: List[Dict[str, Any]], qid2example: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Evaluate predictions against references.

    Args:
        predictions: A list of predictions.
        qid2example: A dictionary of reference with question_id as key.

    Returns:
        Dict[str, float]: A dictionary containing the final scores for time,
        quantity, entity, and overall predictions.
    """
    final_score, score_time, score_num, score_string = get_results(
        predictions, qid2example
    )
    return {
        'score': round(final_score * 100, 2),
        'score_time': round(score_time * 100, 2),
        'score_num': round(score_num * 100, 2),
        'score_string': round(score_string * 100, 2),
    }


def evaluate_infoseek_full(
    predictions: List[List[Dict[str, Any]]],
    qid2examples: List[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    infoseek_score = []
    for pred, qid2example in zip(predictions, qid2examples):
        split_score = evaluate_infoseek(pred, qid2example)
        infoseek_score.append(split_score)
    split_scores = [score['score'] for score in infoseek_score]
    return {
        'final_score': round(harmonic_mean(*split_scores), 2),
        'unseen_question_score': infoseek_score[0],
        'unseen_entity_score': infoseek_score[1],
    }


def evaluate(prediction_path: str, reference_path: str, reference_qtype_path: str, adjust_score: bool) -> Dict[str, Any]:
    """Evaluate predictions against references.

    Args:
        prediction_path: Path to prediction file.
        reference_path: Path to reference file.
        reference_qtype_path: Path to reference question type file.

    Returns:
        Dict[str, Any]: A dictionary containing the final scores for time,
        quantity, entity, and overall predictions.
    """
    predictions = load_jsonl(prediction_path)
    reference = load_jsonl(reference_path)
    reference_qtype = load_jsonl(reference_qtype_path)
    if adjust_score:
        ids = set(x['data_id'] for x in predictions)
        reference = [x for x in reference if x['data_id'] in ids]
        reference_qtype = [x for x in reference_qtype if x['data_id'] in ids]
    qid2example = prepare_qid2example(reference, reference_qtype)
    # split predictions into two splits: unseen_question and unseen_entity
    unseen_question = []
    unseen_entity = []
    for pred in predictions:
        data_id = pred['data_id']
        if data_id in qid2example:
            if qid2example[data_id]['data_split'].endswith('unseen_question'):
                unseen_question.append(pred)
            else:
                unseen_entity.append(pred)
        else:
            pass
    return evaluate_infoseek_full([unseen_question, unseen_entity], [qid2example, qid2example])


def prepare_qid2example(
    reference: List[Dict[str, Any]],
    reference_qtype: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Convert reference to qid2example dictionary."""
    qid2qtype = dict()
    for qtype in reference_qtype:
        qid = qtype["data_id"]
        qid2qtype[qid] = qtype["question_type"]

    qid2example = dict()
    for r in reference:
        qid = r['data_id']
        q_type = qid2qtype[qid].lower()
        if q_type == 'numerical':
            # Process numerical answer:
            # "answer_eval": [{"wikidata": 1.0, "range": [0.9, 1.1]}]
            # --> "answer_eval": [0.9, 1.1]
            if isinstance(r['answer_eval'], list):
                ans_eval = r['answer_eval'][0]['range']
            else:
                ans_eval = r['answer_eval']['range']
            r['answer_eval'] = [str(ans) for ans in ans_eval][:2]

        qid2example[qid] = r
        qid2example[qid]["question_type"] = qid2qtype[qid]
    return qid2example


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of Dict[strionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(ujson.loads(line))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, nargs='+')
    parser.add_argument('--adjust_score', action='store_true')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--reference_path', type=str)
    parser.add_argument('--reference_qtype_path', type=str)
    args = parser.parse_args()

    for i, pred_path in enumerate(args.input_path):
        print(f"=== {pred_path.split('/')[-1]} ===")
        
        merge_data = f'{pred_path}/merge.jsonl'
        if not os.path.exists(merge_data):
            files = [os.path.join(pred_path, f) for f in listdir(pred_path) if isfile(join(pred_path, f)) and ('results' not in f)]
            data = []
            for z in files:
                try:
                    with open(z, 'r') as f:
                        single_file = ujson.load(f)
                    for element in single_file:
                        data.append(element)
                except:
                    print(f'Impossible to open file: {z}')
            
            merge_data = f'{pred_path}/merge.jsonl'
            with open(merge_data, 'w') as f:
                for entry in data:
                    ujson.dump(entry, f)
                    f.write('\n')
                
        result = evaluate(merge_data, args.reference_path, args.reference_qtype_path, args.adjust_score)
        final_score = result["final_score"]
        unseen_question_score = result["unseen_question_score"]["score"]
        unseen_entity_score = result["unseen_entity_score"]["score"]
        print(
            f"FINAL SCORE: {final_score}\n"
            f"UNSEEN QUESTION SCORE: {unseen_question_score}\n"
            f"unseen question score time: {result['unseen_question_score']['score_time']}\n"
            f"unseen question score num: {result['unseen_question_score']['score_num']}\n"
            f"unseen question score string: {result['unseen_question_score']['score_string']}\n"
            f"UNSEEN ENTITY SCORE: {unseen_entity_score}\n"
            f"unseen entity score time: {result['unseen_entity_score']['score_time']}\n"
            f"unseen entity score num: {result['unseen_entity_score']['score_num']}\n"
            f"unseen entity score string: {result['unseen_entity_score']['score_string']}\n"
        )

        meta_result = {}
        meta_result['FINAL SCORE'] = final_score
        meta_result['UNSEEN QUESTION'] = unseen_question_score
        meta_result['unseen question score time'] = result['unseen_question_score']['score_time']
        meta_result['unseen question score num'] = result['unseen_question_score']['score_num']
        meta_result['unseen question score string'] = result['unseen_question_score']['score_string']
        meta_result['UNSEEN ENTITY SCORE'] = unseen_entity_score
        meta_result['unseen entity score time'] = result['unseen_entity_score']['score_time']
        meta_result['unseen entity score num'] = result['unseen_entity_score']['score_num']
        meta_result['unseen entity score string'] = result['unseen_entity_score']['score_string']
        
        with open(f'{pred_path}/results.json', 'w') as f:
            ujson.dump(meta_result, f)
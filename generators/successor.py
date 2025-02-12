from generators.generator import DataGenerator

import random
import torch
from typing import List, Tuple

DEFAULT_PROMPT = """
You are a great mathematician. You will be tasked with generating a series of numbers. The series will be an increasing series with constant difference 1. 
You will be given the start of the sequence (the first number) and your task will be to continue generating the next elements indefinitely.
Do not generate any words. You can only generate digits and numbers from the series.

"""

DEFAULT_FEW_SHOT_EXAMPLES = """
Here are some examples:

H3 K8 U1 D5 G9 F1 HKKHDFG=3883519
AB CD EF GH IJ KL ACEGIK=BDFHJL

Use them to solve the following task:

"""

"""
Class provides functionality for generating data samples for successor task.
Example samples:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
234 235 236 237 238 239 240
2 3 4 5 6 7 8 9 10 11 12 13
50 51 52 53 54

The only parameters of this type of tasks are:
- the range of the starting number
- the length of the series

Each generated sample comes with a corresponding ground truth for attention scores.
These attention labels are used for inspection into learned weights, i.e. to see
whether the attention layer learned to attend correctly to the correct tokens.
"""
class Successor(DataGenerator):
    """Initiate the data generator class with a seed for reproducibility/comparability

    Args:
        DataGenerator
    """
    def __init__(self, seed: int, 
                 tokenizer, 
                 prompt=DEFAULT_PROMPT, 
                 few_shot_examples=DEFAULT_FEW_SHOT_EXAMPLES,
                 use_prompt=False,
                 use_few_shot=False,
                 start_number_range=(1, 100),
                 length=(1,100),
                 ignore_index=-100.0,
                 generate_attn_labels=True
                 ):
        
        self.random = random.Random(seed)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.few_shot_examples = few_shot_examples
        self.use_prompt = use_prompt
        self.use_few_shot = use_few_shot
        self.length = length
        self.start_number_range = start_number_range
        self.ignore_index = ignore_index
        self.generate_attn_labels = generate_attn_labels


    def generate_attention_labels(self, sample: List[str], input_ids: List[List[int]], prompt_length: int, preprompt_length: int, ignore_index: int) -> torch.Tensor:
        # Samples are of form: A1B2C4 ACB=142

        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100.0
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:prompt_length-1, :] = ignore_index

        last_whitespace_idx = preprompt_length
        current_whitespace_idx = preprompt_length
        # The model should always attend to the whole previous number and nothing else
        for i in range(prompt_length-1, len(input_ids)):
            if sample[i] == " ":
                last_whitespace_idx = current_whitespace_idx

            attn_labels[i, last_whitespace_idx:i+1] = 1.0
            
            if sample[i] == " ":
                current_whitespace_idx = i
        
        return attn_labels


    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_number = self.random.randint(*self.start_number_range)
        length = self.random.randint(*self.length)

        text_sample = ""
        for i in range(length):
            text_sample += str(start_number + i) + " "

        prompt_length = len(list(str(start_number))) + 1

        # Encode every character into single-char token
        tokenized = []
        for char in text_sample:
            tokenized += self.tokenizer.encode(char)

        preprompt_length = 0
        if self.use_few_shot:
            tokenized_few_shot_examples = self.tokenizer.encode(self.few_shot_examples)
            tokenized = tokenized_few_shot_examples + tokenized
            prompt_length += len(tokenized_few_shot_examples)
            preprompt_length += len(tokenized_few_shot_examples)
            text_sample = self.few_shot_examples + text_sample
        if self.use_prompt:
            tokenized_prompt = self.tokenizer.encode(self.prompt)
            tokenized = tokenized_prompt + tokenized
            prompt_length += len(tokenized_prompt)
            preprompt_length += (len(tokenized_prompt))
            text_sample = self.prompt + text_sample

        input_ids = torch.LongTensor(tokenized)
        target_ids = input_ids.clone()
        
        # Mask the prompt part in the target_ids
        # But leave out the last token of the prompt
        # The last token needs to "learn" to predict the next token
        # So it has to be learning, it cannot be masked
        target_ids[:prompt_length-1] = self.ignore_index

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        attn_labels = None
        if self.generate_attn_labels:
            attn_labels = self.generate_attention_labels(text_sample, input_ids, prompt_length, preprompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


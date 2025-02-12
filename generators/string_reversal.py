from generators.generator import DataGenerator

import random
from typing import List, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEFAULT_PROMPT = """
You are an expert in solving tasks requiring manipulation of symbols. 
Given an arbitrary input string of symbols, your task will be to produce an exact reversed copy of these symbols.
Do not generate any words. You can only generate the symbols of the reversed string of symbols and nothing more.
"""

DEFAULT_FEW_SHOT_EXAMPLES = """
Here are some examples:

123456789=987654321
abhdsifhidasf=fsadihfisdhba
1=1
9j7g2k0b8h=h8b0k2g7j9

Use them to solve the following task:
"""


"""
Class provides functionality for generating data samples for the string reversal task.
Example samples:
123456789=987654321 (only numbers)
dh13h82hj283j23H=H32j382jh28h31hd (alphanumeric case-sensitive characters)

The only parameters of this type of tasks are:
- the character set from which strings are sampled
- the length of the strings

Each generated sample comes with a corresponding ground truth for attention scores.
These attention labels are used for inspection into learned weights, i.e. to see
whether the attention layer learned to attend correctly to the correct tokens.
"""
class StringReversal(DataGenerator):
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
                 length=(1,100),
                 ignore_index=-100,
                 generate_attn_labels=True,
                 eos_token_id=0):
        self.random = random.Random(seed)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.few_shot_examples = few_shot_examples
        self.use_prompt = use_prompt
        self.use_few_shot = use_few_shot
        self.length = length
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.generate_attn_labels = generate_attn_labels

    def generate_attention_labels(self, text_sample: str, input_ids, preprompt_length: int, prompt_length: int, ignore_index: int) -> torch.Tensor:
        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:prompt_length-1, :] = ignore_index
        
        # Set the attention labels such that each token attends to its corresponding position in the reversed string
        # Add additional -1 to indexes to compensate for the shift in target_ids
        for i in range(prompt_length, size-1):
            attn_labels[i, prompt_length - 2 - i] = 1.0

        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        random_string = ''.join(random.choices(list(charset), k=length))
        reversed_string = random_string[::-1]
        text_sample = random_string + "=" + reversed_string
        
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char)
        tokenized_sample += [self.eos_token_id]
        
        prompt_length = len(random_string) + 1
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

        input_ids = torch.LongTensor(tokenized_sample)
        target_ids = input_ids.clone()
        
        # Mask the prompt part in the target_ids
        # But leave out the last "=" token so it can learn to predict the first symbol
        target_ids[:prompt_length-1] = ignore_index

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        attn_labels = None
        if self.generate_attention_labels:
            attn_labels = self.generate_attention_labels(text_sample, tokenized_sample[:-1], preprompt_length, prompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


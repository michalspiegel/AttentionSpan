from generators.generator import DataGenerator

import random
from typing import List, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEFAULT_PROMPT = """
You are an expert in solving arithmetic tasks. You will be tasked with computing the sum of input numbers.
The number are all represented with their least significant digit first.
Sum these number using the standard addition algorithm. When predicting each digit of the result, always sum the corresponding digits of each number and remember the carry.
Do not generate any words. You can only generate number and nothing else.
"""

DEFAULT_FEW_SHOT_EXAMPLES = """
Here are some examples:

1+1=2
1240+4335+3440=8916
999+999=8991

Use them to solve the following task:
"""


"""
Class provides functionality for generating data samples for the long addition task.
Example samples:
1+1=2
1240+4335+3440=8916
999+999=8991

The only parameters of this type of tasks are:
- the length of the numbers
- the number of the numbers
- whether to pad with zeros from right

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
                 length=(1,20),
                 num_of_nums=(2,3),
                 pad_with_zeros=True,
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
        self.num_of_nums = num_of_nums
        self.pad_with_zeros = pad_with_zeros
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.generate_attn_labels = generate_attn_labels

    def generate_attention_labels(self, text_sample: str, 
                                  input_ids, 
                                  num_of_nums: int, 
                                  max_length: int, 
                                  preprompt_length: int, 
                                  prompt_length: int, 
                                  ignore_index: int) -> torch.Tensor:
        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:prompt_length-1, :] = ignore_index
        
        for i in range(prompt_length, size-1):
            for j in range(num_of_nums):
                attn_labels[i, preprompt_length+j*(max_length+1)+i] = 1.0

        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        text_sample = ""
        numbers = []
        num_of_nums = self.random.randint(*self.num_of_nums)
        for _ in range(num_of_nums):
            num_digits = self.random.randint(*self.length)
            num = self.random.randint(10**num_digits-1, 10**num_digits)
            numbers.append(num)
        sum_of_nums = sum(numbers)
        max_length = max([len(str(num)) for num in numbers]) + 1
        
        # Stringify and reverse digits
        str_sum_of_nums = str(sum_of_nums)[::-1]
        str_numbers = []
        for i in range(len(numbers)-1):
            str_numbers.append(str(numbers[i])[::-1])
        
        if self.pad_with_zeros:
            str_sum_of_nums = str_sum_of_nums.ljust(max_length, "0")
            str_numbers = [number.ljust(max_length, "0") for number in str_numbers]

        text_sample = "+".join(str_numbers) + "=" + str_sum_of_nums
        
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char)
        tokenized_sample += [self.eos_token_id]
        
        prompt_length = num_of_nums*max_length + num_of_nums
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
        target_ids[:prompt_length-1] = self.ignore_index

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        attn_labels = None
        if self.generate_attention_labels:
            attn_labels = self.generate_attention_labels(text_sample, tokenized_sample[:-1], num_of_nums, max_length, preprompt_length, prompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


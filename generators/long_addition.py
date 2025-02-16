from generators.generator import DataGenerator

import random
from typing import List, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

CHAT_TEMPLATE = [
  {"role": "user", "content": "{tokens}{tokens}"},
  {"role": "assistant", "content": "{tokens}"},
]

DEFAULT_INSTRUCTION = """
You are an expert in solving arithmetic tasks. You will be tasked with computing the sum of input numbers.
The number are all represented with their least significant digit first.
Sum these number using the standard addition algorithm. When predicting each digit of the result, always sum the corresponding digits of each number and remember the carry.
Do not generate any words. You can only generate number and nothing else.
"""

DEFAULT_FEW_SHOT_PROMPT = (
"\nHere are some examples:\n",
"\nUse them to solve the following task:\n"
)

DEFAULT_FEW_SHOT_SAMPLES = [
"1+1=2",
"1240+4335+3440=8916",
"999+999=8991",
]


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
class LongAdditionGenerator(DataGenerator):
    """Initiate the data generator class with a seed for reproducibility/comparability

    Args:
        DataGenerator
    """
    def __init__(self, seed: int, 
                 tokenizer,
                 instruction=DEFAULT_INSTRUCTION, 
                 few_shot_prompt=DEFAULT_FEW_SHOT_PROMPT,
                 few_shot_samples=DEFAULT_FEW_SHOT_SAMPLES,
                 use_instruction=False,
                 use_few_shot=False,
                 length=(1,20),
                 num_of_nums=(2,3),
                 pad_with_zeros=True,
                 ignore_index=-100,
                 generate_attn_labels=True,
                 eos_token_id=0,
                 apply_chat_template=False):
        self.random = random.Random(seed)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.few_shot_samples = few_shot_samples
        self.use_instruction = use_instruction
        self.use_few_shot = use_few_shot
        self.length = length
        self.num_of_nums = num_of_nums
        self.pad_with_zeros = pad_with_zeros
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.generate_attn_labels = generate_attn_labels
        self.apply_chat_template = apply_chat_template

    def generate_attention_labels(self, text_sample: str, 
                                  input_ids,
                                  generation_prompt_length: int,
                                  num_of_nums: int, 
                                  max_length: int, 
                                  instruction_length: int, 
                                  prompt_length: int, 
                                  ignore_index: int) -> torch.Tensor:
        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:instruction_length+prompt_length+generation_prompt_length-1, :] = ignore_index
        attn_labels[:, :instruction_length] = ignore_index

        for i in range(max_length):
            for j in range(num_of_nums):
                attn_labels[instruction_length+prompt_length+generation_prompt_length+i-1, instruction_length+j*(max_length+1)+i] = 1.0

        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        text_sample = ""
        numbers = []
        num_of_nums = max(self.random.randint(*self.num_of_nums), 2)
        for _ in range(num_of_nums):
            num_digits = self.random.randint(*self.length)
            num = self.random.randint(10**(num_digits-1), 10**num_digits)
            numbers.append(num)
        sum_of_nums = sum(numbers)
        max_length = max([len(str(num)) for num in numbers]) + 1
        
        # Stringify and reverse digits
        str_sum_of_nums = str(sum_of_nums)[::-1]
        str_numbers = []
        for i in range(len(numbers)):
            str_numbers.append(str(numbers[i])[::-1])
        


        if self.pad_with_zeros:
            str_sum_of_nums = str_sum_of_nums.ljust(max_length, "0")
            str_numbers = [number.ljust(max_length, "0") for number in str_numbers]

        text_sample = "+".join(str_numbers) + "=" + str_sum_of_nums
        
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char, add_special_tokens=False)
        
        prompt_length = len("+".join(str_numbers) + "=")
        
        text_instruction = ""
        tokenized_instruction = []
        if self.use_instruction:
            tokenized_instruction += self.tokenizer.encode(self.instruction, add_special_tokens=False)
            text_instruction += self.instruction
        if self.use_few_shot:
            tokenized_instruction += self.tokenizer.encode(self.few_shot_prompt[0], add_special_tokens=False)
            for sample in self.few_shot_samples:
                for char in sample:
                    tokenized_instruction += self.tokenizer.encode(char, add_special_tokens=False)
                tokenized_instruction += self.tokenizer.encode("\n", add_special_tokens=False)
            tokenized_instruction += self.tokenizer.encode(self.few_shot_prompt[1], add_special_tokens=False)
            text_instruction += self.few_shot_prompt[0] + "\n".join(self.few_shot_samples) + self.few_shot_prompt[1]

        if self.apply_chat_template:
            template = self.tokenizer.apply_chat_template(CHAT_TEMPLATE, tokenize=False)
            template_split = template.split("{tokens}")
            tokens = self.tokenizer.encode(template_split[0])
            tokens += tokenized_instruction
            tokens += self.tokenizer.encode(template_split[1], add_special_tokens=False)
            instruction_length = len(tokens)
            tokens += tokenized_sample[:prompt_length]
            generation_prompt=self.tokenizer.encode(template_split[2], add_special_tokens=False)
            tokens += generation_prompt
            tokens += tokenized_sample[prompt_length:]
            tokens += self.tokenizer.encode(template_split[3], add_special_tokens=False)
            text_sample = template_split[0] + text_instruction + template_split[1] + text_sample[:prompt_length] + template_split[2] + text_sample[prompt_length:] + template_split[3]
            tokenized_sample = tokens    
            generation_prompt_length = len(generation_prompt)
        else:
            tokenized_sample = [self.tokenizer.bos_token_id] + tokenized_instruction + tokenized_sample + [self.tokenizer.eos_token_id]
            text_sample = text_instruction + text_sample
            instruction_length = len(tokenized_instruction) + 1
            generation_prompt_length = 0

        input_ids = torch.LongTensor(tokenized_sample)
        target_ids = input_ids.clone()
        
        # Mask the prompt part in the target_ids
        # But leave out the last "=" token so it can learn to predict the first symbol
        target_ids[:instruction_length+prompt_length+generation_prompt_length] = self.ignore_index

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        attn_labels = None
        if self.generate_attention_labels:
            attn_labels = self.generate_attention_labels(text_sample, tokenized_sample[:-1], generation_prompt_length, num_of_nums, max_length, instruction_length, prompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


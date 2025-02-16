from generators.generator import DataGenerator

import random
import torch
from typing import List, Tuple

CHAT_TEMPLATE = [
  {"role": "user", "content": "{tokens}{tokens}"},
  {"role": "assistant", "content": "{tokens}"},
]

DEFAULT_INSTRUCTION= """
You are a great mathematician. You will be tasked with generating a series of numbers. The series will be an increasing series with constant difference 1. 
You will be given the start of the sequence (the first number) and your task will be to continue generating the next elements indefinitely.
Do not generate any words. You can only generate digits and numbers from the series.

"""

DEFAULT_FEW_SHOT_PROMPT = (
"Here are some examples:\n",
"Use them to solve the following task:\n"
)

DEFAULT_FEW_SHOT_SAMPLES = [
"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17",
"234 235 236 237 238 239 240",
"2 3 4 5 6 7 8 9 10 11 12 13",
"50 51 52 53 54"
]


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
class SuccessorGenerator(DataGenerator):
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
                 start_number_range=(1, 100),
                 length=(1,100),
                 ignore_index=-100.0,
                 generate_attn_labels=True,
                 apply_chat_template=False
                 ):
        
        self.random = random.Random(seed)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.few_shot_samples = few_shot_samples
        self.use_instruction = use_instruction
        self.use_few_shot = use_few_shot
        self.length = length
        self.start_number_range = start_number_range
        self.ignore_index = ignore_index
        self.generate_attn_labels = generate_attn_labels
        self.apply_chat_template = apply_chat_template


    def generate_attention_labels(self, sample: List[str], input_ids: List[List[int]], generation_prompt_length: int, prompt_length: int, instruction_length: int, ignore_index: int) -> torch.Tensor:
        # Samples are of form: A1B2C4 ACB=142

        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        # Set the values of tokens that are in the prompt to -100.0
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:instruction_length+prompt_length+generation_prompt_length-1, :] = ignore_index
        attn_labels[:, :instruction_length] = ignore_index

        last_whitespace_idx = instruction_length-1
        current_whitespace_idx = instruction_length-1
        whitespace_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
        # The model should always attend to the whole previous number and nothing else
        for i in range(instruction_length+prompt_length+generation_prompt_length-1, size-1):
            if i == instruction_length+prompt_length+generation_prompt_length-1 or input_ids[i].item() == whitespace_token_id:
                last_whitespace_idx = current_whitespace_idx

            attn_labels[i, last_whitespace_idx+1:i+1] = 1.0
            if last_whitespace_idx <= instruction_length:
                attn_labels[i, instruction_length+prompt_length:instruction_length+prompt_length+generation_prompt_length] = 0.0
            
            if i == instruction_length+prompt_length+generation_prompt_length-1 or input_ids[i].item() == whitespace_token_id:
                current_whitespace_idx = i
        
        return attn_labels


    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_number = self.random.randint(*self.start_number_range)
        length = self.random.randint(*self.length)

        text_sample = ""
        for i in range(length):
            text_sample += str(start_number + i) + " "

        prompt_length = len(str(start_number)) + 1

        # Encode every character into single-char token
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char, add_special_tokens=False)

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
            text_sample = template_split[0] + text_instruction + template_split[1] + text_sample[:prompt_length] + template_split[2] + text_sample[prompt_length:]
            tokenized_sample = tokens    
            generation_prompt_length = len(generation_prompt)
        else:
            tokenized_sample = [self.tokenizer.bos_token_id] + tokenized_instruction + tokenized_sample
            text_sample = text_instruction + text_sample
            instruction_length = len(tokenized_instruction) + 1
            generation_prompt_length = 0

        input_ids = torch.LongTensor(tokenized_sample)
        target_ids = input_ids.clone()
        
        # Mask the prompt part in the target_ids
        target_ids[:instruction_length+prompt_length+generation_prompt_length] = self.ignore_index

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]

        attn_labels = None
        if self.generate_attn_labels:
            attn_labels = self.generate_attention_labels(text_sample, input_ids, generation_prompt_length, prompt_length, instruction_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


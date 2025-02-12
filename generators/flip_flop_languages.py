from generators.generator import DataGenerator

import random
from typing import List, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEFAULT_PROMPT = """
You are an expert in solving tasks requiring manipulation of symbols. Your task will be to evaluate a string representing a register.
The string is composed of commands to the register: write, read, ignore, flip. Each is represented by a single character (w, r, i, f)
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
Class provides functionality for generating data samples for the flip flop language modeling task (simulating 1-10 1-bit registers).
Each sample always ends with a read command to test the model about the contents of a particular register.
Example samples:

w11i11f10r10f10r11


The only parameters of this type of tasks are:
- how many register to use
- the length of the strings
- whether to use flip commands

Each generated sample comes with a corresponding ground truth for attention scores.
These attention labels are used for inspection into learned weights, i.e. to see
whether the attention layer learned to attend correctly to the correct tokens.
"""
class FlipFlopLanguageModeling(DataGenerator):
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
                 use_flip=True,
                 num_registers=(1,10),
                 ignore_index=-100,
                 generate_attn_labels=True
                 ):
        self.random = random.Random(seed)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.few_shot_examples = few_shot_examples
        self.use_prompt = use_prompt
        self.use_few_shot = use_few_shot
        self.length = length
        self.use_flip = use_flip
        self.num_registers = num_registers
        self.ignore_index = ignore_index
        self.generate_attn_labels = generate_attn_labels

    def generate_attention_labels(self, text_sample: str, input_ids, preprompt_length: int, prompt_length: int, ignore_index: int) -> torch.Tensor:
        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:prompt_length-1, :] = ignore_index
        
        register_idx = text_sample[-1]        
        for i in range(len(text_sample)-1):
            if text_sample[i] in set("w", "f", "r") and text_sample[i+1] == register_idx:
                attn_labels[prompt_length, i:i+1] = 1.0

        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
       
        text_sample = ""
        num_registers = self.random.randint(*self.num_registers)
        registers = [0]*num_registers
        for _ in range(self.random.randint(*self.length)-1):
            if self.use_flip:
                command = self.random.choice(["w", "r", "i", "f"])
            else:
                command += self.random.choice(["w", "r", "i"])
            registers_idx = str(self.random.randint(0, num_registers-1))
            value = self.random.choice(["0", "1"])
            text_sample += command + registers_idx + value
            
            if command == "w":
                registers[int(registers_idx)] = value
            elif command == "f":
                registers[int(registers_idx)] = "0" if registers[int(registers_idx))] == "1" else "1"
        
        read_idx = self.random.randint(0, num_registers-1)
        text_sample += "r" + str(read_idx) + registers[read_idx]
        
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char)
        
        prompt_length = len(text_sample) - 1
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
            attn_labels = self.generate_attention_labels(text_sample, tokenized_sample[:-1], preprompt_length, prompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


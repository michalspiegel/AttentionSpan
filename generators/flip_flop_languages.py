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
You are an expert in solving tasks requiring manipulation of symbols. Your task will be to evaluate a string representing a register.
The string is composed of commands to the register: write, read, ignore, flip. Each is represented by a single character (w, r, i, f)
Do not generate any words. You can only generate the symbols of the reversed string of symbols and nothing more.
"""

DEFAULT_FEW_SHOT_PROMPT = (
"\nHere are some examples:\n",
"\nUse them to solve the following task:\n"
)

DEFAULT_FEW_SHOT_SAMPLES = [
"w11i11f10r10f10r11"
"i11i11i11i11i11i11r10"
]


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
class FlipFlopGenerator(DataGenerator):
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
                 length=(1,100),
                 use_flip=True,
                 num_registers=(1,10),
                 ignore_index=-100,
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
        self.use_flip = use_flip
        self.num_registers = num_registers
        self.ignore_index = ignore_index
        self.generate_attn_labels = generate_attn_labels
        self.apply_chat_template = apply_chat_template


    def generate_attention_labels(self, text_sample: str, input_ids, generation_prompt_length: int, instruction_length: int, prompt_length: int, ignore_index: int) -> torch.Tensor:
        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:instruction_length+prompt_length+generation_prompt_length-1, :] = ignore_index
        attn_labels[:, :instruction_length] = ignore_index
        
        # Attend the register number
        attn_labels[instruction_length+prompt_length+generation_prompt_length-1, instruction_length+prompt_length-1] = 1.0
        register_token_id = input_ids[instruction_length+prompt_length-1]
        
        write_token_id = self.tokenizer.encode("w", add_special_tokens=False)[0]
        flip_token_id = self.tokenizer.encode("f", add_special_tokens=False)[0]
        
        # Go backward and stop when you find the last write
        for i in range(instruction_length+prompt_length-2,instruction_length+2, -3):
            if input_ids[i] == write_token_id and input_ids[i+1] == register_token_id:
                attn_labels[instruction_length+prompt_length+generation_prompt_length-1, i:i+3] = 1.0
                break
            elif input_ids[i] == flip_token_id and input_ids[i+1] == register_token_id:
                attn_labels[instruction_length+prompt_length+generation_prompt_length-1, i:i+2] = 1.0

        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
       
        text_sample = ""
        num_registers = self.random.randint(*self.num_registers)
        registers = [0]*num_registers
        for _ in range(self.random.randint(*self.length)-1):
            if self.use_flip:
                command = self.random.choice(["w", "i", "f"])
            else:
                command += self.random.choice(["w", "i"])
            registers_idx = str(self.random.randint(0, num_registers-1))
            value = self.random.choice(["0", "1"])
            text_sample += command + registers_idx + value
            if command == "w":
                registers[int(registers_idx)] = value
            elif command == "f":
                registers[int(registers_idx)] = "0" if registers[int(registers_idx)] == "1" else "1"
        
        read_idx = self.random.randint(0, num_registers-1)
        text_sample += "r" + str(read_idx) + str(registers[read_idx])
        
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char, add_special_tokens=False)
        
        prompt_length = len(text_sample) - 1
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
            attn_labels = self.generate_attention_labels(text_sample, tokenized_sample[:-1], generation_prompt_length, instruction_length, prompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


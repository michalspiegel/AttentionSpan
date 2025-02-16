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
You are an expert in solving arithmetic tasks. You will be tasked with multiplying two input numbers. Input number are always reversed and padded with zeros from the right.
Multiply these numbers using the standard multiplication algorithm.
Start from the first digit of the second number and multiply this digit with every digit of the second number (starting from left). Remember the carry.
Do not generate any words. You can only generate the digits, symbol + and symbol =.
"""

DEFAULT_FEW_SHOT_PROMPT = (
"\nHere are some examples:\n",
"\nUse them to solve the following task:\n"
)

DEFAULT_FEW_SHOT_SAMPLES = [
"10*10=10+00=10",
"9900*9900=1980+0198+0000+0000=1089"
]


"""
Class provides functionality for generating data samples for the long multiplication task.
By default, the generator can also explicitly generate the carry in the string of intermediate computations. 
The carry is always behind each digit in brackets. 1(8) symbolizes digit 1 with carry 8 to the next digit.

Example samples:
12300*12000=12300+02460+00000+00000+00000=14760
9900*9900=1980+0198+0000+0000=1089

(This is an example with generating the carry explicitly after each digit of the intermediate products)
TODO: This is not yet implemented!!!
9900*9900=1(8)9(8)8(0)0+0(0)1(8)9(0)8+0(0)0(0)0(0)0+0(0)0(0)0(0)0=1(0)0(1)8(1)9  (example with generating carry tokens in the intermediate computations)

The only parameters of this type of tasks are:
- the number of digits of the numbers
- whether to pad with zeros from right
- whether to explicitly generate carry tokens

Each generated sample comes with a corresponding ground truth for attention scores.
These attention labels are used for inspection into learned weights, i.e. to see
whether the attention layer learned to attend correctly to the correct tokens.
"""
class LongMultiplicationGenerator(DataGenerator):
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
                 pad_with_zeros=True,
                 generate_carry=True,
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
        self.pad_with_zeros = pad_with_zeros
        self.generate_carry = generate_carry
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.generate_attn_labels = generate_attn_labels
        self.apply_chat_template = apply_chat_template

    def generate_attention_labels(self, text_sample: str, 
                                  input_ids,
                                  generation_prompt_length: int,
                                  max_length: int, 
                                  instruction_length: int, 
                                  prompt_length: int, 
                                  ignore_index: int) -> torch.Tensor:
        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens
        attn_labels[:instruction_length+prompt_length-1, :] = ignore_index
        attn_labels[:, :instruction_length] = ignore_index

        number_of_intermediate_products = max_length
        for i in range(number_of_intermediate_products):
            # For each digit in the intermediate product
            for j in range(max_length):
                # Attend to the i-th digit of the 2nd number and j-th digit of the 1st number
                attn_labels[instruction_length+generation_prompt_length+prompt_length + i*(max_length+1) + j - 1, instruction_length+max_length+1+i] = 1.0
                attn_labels[instruction_length+generation_prompt_length+prompt_length + i*(max_length+1) + j - 1, instruction_length+j] = 1.0
        
        # REMOVED: The model does not have to necessary attend to its own outputs
        ##For each digit of the final result
        #intermediate_computation_length = instruction_length + prompt_length + max_length*max_length + max_length - 1
        #for i in range(max_length):
        #    for j in range(number_of_intermediate_products):
        #        attn_labels[intermediate_computation_length + i, instruction_length+prompt_length + j*(max_length+1) + i] = 1.0
        

        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        text_sample = ""
        num1_length = self.random.randint(*self.length)
        num2_length = self.random.randint(*self.length)
        num1 = self.random.randint(10**(num1_length-1), 10**num1_length)
        num2 = self.random.randint(10**(num2_length-1), 10**num2_length)
        # num2 must be always smaller or equal
        if num1 < num2:
            num1, num2 = num2, num1
            num1_length, num2_length = num2_length, num1_length
        # Estimate the max length of the product
        max_length = num1_length * 2
        product = num1*num2
        intermediate_products = []
        
        # Compute the intermediate products of the standard multiplication algorithm
        for idx, digit in enumerate(reversed(str(num2))):
            intermediate_products.append(num1 * int(digit) * 10**idx)
        
        # Stringify and reverse
        str_num1 = str(num1)[::-1]
        str_num2 = str(num2)[::-1]
        str_product = str(product)[::-1]
        str_intermediate_products = [str(product)[::-1] for product in intermediate_products]


        if self.pad_with_zeros:
            str_num1 = str_num1.ljust(max_length, "0")
            str_num2 = str_num2.ljust(max_length, "0")
            str_product = str_product.ljust(max_length, "0")
            str_intermediate_products = [str_product.ljust(max_length, "0") for str_product in str_intermediate_products]
            if len(str_intermediate_products) < max_length: 
                str_intermediate_products += ["0"*max_length]*(max_length - len(str_intermediate_products))

        text_sample = str_num1 + "*" + str_num2 + "=" + "+".join(str_intermediate_products) + "=" + str_product
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char, add_special_tokens=False)
        
        prompt_length = len(str_num1) + len(str_num2) + 2
        
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
            attn_labels = self.generate_attention_labels(text_sample, tokenized_sample[:-1], generation_prompt_length, max_length, instruction_length, prompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels


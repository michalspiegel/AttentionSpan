from generator import DataGenerator

import random
import torch
from typing import List, Tuple

CHAT_TEMPLATE = [
  {"role": "user", "content": "{tokens}{tokens}"},
  {"role": "assistant", "content": "{tokens}"},
]

DEFAULT_INSTRUCTION = """
You are a great problem solver excellent in tasks requiring manipulation of symbols. You will be tasked with encoding a sequence of characters given a set of encodings. 
An encoding is a tuple of characters, e.g. A1, where A is the character to be encoded, and 1 is its encoding. For example, given some string "A", its encoding is a sequence "1". 
Given a set of these encodings and an input string, encode this string.
Do not generate any words. You can only generate characters from the set of encodings.
"""

DEFAULT_FEW_SHOT_PROMPT = [
"Here are some examples:\n",
"Use them to solve the following task:\n"
]

DEFAULT_FEW_SHOT_SAMPLES = (
"H3K8U1D5G9F1 HKKHDFG=3883519",
"ABCDEFGHIJKL ACEGIK=BDFHJL"
)

"""
Class provides functionality for generating data samples for the boolean formula evaluation task.
Example sample:
B1E0D1A1C0  ABBEDACABCD=11101101101

H3K8U1D5G9F1 HKKHDFG=3883519


The only parameters of this type of tasks are:
- the character set from which to sample encodings
- the length of encodings and the length of the strings to be encrypted

Each generated sample comes with a corresponding ground truth for attention scores.
These attention labels are used for inspection into learned weights, i.e. to see
whether the attention layer learned to attend correctly to the correct tokens.
"""
class ValueAssignmentGenerator(DataGenerator):
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
                 charset_from=set("qwertzuiopasdfghjklyxcvbnmQWERTZUIOPASDFGHJKLYXCVBNM1234567890"),
                 charset_to=set("qwertzuiopasdfghjklyxcvbnmQWERTZUIOPASDFGHJKLYXCVBNM1234567890"),
                 length=(1,100),
                 encoding_length=(1,100),
                 ignore_index=-100,
                 eos_token_id=0,
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
        self.charset_from = charset_from
        self.charset_to = charset_to
        self.encoding_length = encoding_length
        self.length = length
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.generate_attn_labels = generate_attn_labels
        self.apply_chat_template = apply_chat_template

    def generate_attention_labels(self, sample: List[str], 
                                  input_ids: List[List[int]],
                                  generation_prompt_length: int,
                                  assignment_length: int, 
                                  input_string_length: int, 
                                  prompt_length: int, 
                                  instruction_length: int, 
                                  ignore_index: int) -> torch.Tensor:

        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100.
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens.
        # The last token of the prompt is predicting the first token, that is why we do not ignore it.
        attn_labels[:instruction_length+prompt_length+generation_prompt_length-1, :] = ignore_index
        attn_labels[:, :instruction_length] = ignore_index
        #For every token of the encoded message
        for i in range(instruction_length+prompt_length+generation_prompt_length-1, instruction_length+prompt_length+generation_prompt_length+input_string_length-1):
            # Attend to the corresponding input symbol in the input string
            attn_labels[i, i-input_string_length-generation_prompt_length] = 1.0
            input_symbol = input_ids[i-input_string_length-generation_prompt_length]
            # Search for the correct assignment/encoding tuple
            for j in range(instruction_length, instruction_length+assignment_length, 2):
                if input_ids[j] == input_symbol:
                    attn_labels[i, j:j+2] = 1.0
        
        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding_length = self.random.randint(*self.encoding_length)
        charset_from_subset = self.random.sample(self.charset_from, k=encoding_length)
        charset_to_subset = self.random.sample(self.charset_to, k=encoding_length)
        random_assignment = {char:self.random.choice(charset_to_subset) for char in self.random.sample(charset_from_subset, k=encoding_length)}
        random_input = self.random.choices(charset_from_subset, k=self.random.randint(*self.length))
        encoded_input = [random_assignment[char] for char in random_input]
        
        random_input_string = "".join(random_input)
        encoded_input_string = "".join(encoded_input)
        random_assignment_prompt = [char+random_assignment[char] for char in random_assignment]
        random_assign_prompt_string = "".join(random_assignment_prompt)

        text_sample = random_assign_prompt_string + " " + random_input_string + "=" + encoded_input_string
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char, add_special_tokens=False)
        
        prompt_length = len(random_assign_prompt_string) + len(random_input_string) + 2
        assignment_length = len(random_assign_prompt_string)
        input_string_length = len(random_input_string)

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
        # But leave out the last token of the prompt
        # The last token needs to "learn" to predict the next token
        # So it has to be learning, it cannot be masked
        target_ids[:instruction_length+prompt_length+generation_prompt_length] = self.ignore_index

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        attn_labels = None
        if self.generate_attn_labels:
            attn_labels = self.generate_attention_labels(text_sample, input_ids, generation_prompt_length, assignment_length, input_string_length,  prompt_length, instruction_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels
    

from generator import DataGenerator

import random
import torch
from typing import List, Tuple

DEFAULT_PROMPT = """
You are a great problem solver excellent in tasks requiring manipulation of symbols. You will be tasked with encoding a sequence of characters given a set of encodings. 
An encoding is a tuple of characters, e.g. A1, where A is the character to be encoded, and 1 is its encoding. For example, given some string "A", its encoding is a sequence "1". 
Given a set of these encodings and an input string, encode this string.
Do not generate any words. You can only generate characters from the set of encodings.
"""

DEFAULT_FEW_SHOT_EXAMPLES = """
Here are some examples:

H3K8U1D5G9F1 HKKHDFG=3883519
ABCDEFGHIJKL ACEGIK=BDFHJL

Use them to solve the following task:
"""

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
class ValueAssignment(DataGenerator):
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
                 charset_from=set("ABCDEFGHIJKLMNOPRST"),
                 charset_to=set("0123456789"),
                 length=(1,100),
                 ignore_index=-100,
                 eos_token_id=0,
                 generate_attn_labels=True
                 ):
        
        self.random.Random(seed)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.few_shot_examples = few_shot_examples
        self.use_prompt = use_prompt
        self.use_few_shot = use_few_shot
        self.charset_from = charset_from
        self.charset_to = charset_to
        self.length = length
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id
        self.generate_attention_labels = generate_attn_labels

    def generate_attention_labels(self, sample: List[str], 
                                  input_ids: List[List[int]], 
                                  assignment_length: int, 
                                  input_string_length: int, 
                                  prompt_length: int, 
                                  preprompt_length: int, 
                                  ignore_index: int) -> torch.Tensor:

        size = len(input_ids)
        attn_labels = torch.zeros((size, size), dtype=torch.int)
        
        # Set the values of tokens that are in the prompt to -100.
        # For these tokens we have no ground truth of attention
        # because the models is not supposed to predict the prompt tokens.
        # The last token of the prompt is predicting the first token, that is why we do not ignore it.
        attn_labels[:prompt_length-1, :] = ignore_index

        for i in range(prompt_length-1, size):
            # Attend to the corresponding input symbol in the input string
            attn_labels[i, i-input_string_length-1] = 1.0
            input_symbol = input_ids[i-input_string_length-1]
            # Search for the correct assignment/encoding tuple
            for j in range(preprompt_length, preprompt_length+assignment_length):
                if input_ids[j] == input_symbol:
                    attn_labels[i, j:j+1] = 1.0

        
        
        return attn_labels

    def generate_sample(self) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        random_assignment = {char:self.random.choice(self.charset_to) for char in self.random.shuffle(self.charset_from)}
        random_input = self.random.choices(self.charset_from, k=self.random.randint(self.length))
        encoded_input = [random_assignment[char] for char in random_input]
        
        random_input_string = "".join(random_input)
        encoded_input_string = "".join(encoded_input)
        random_assignment_prompt = [char+random_assignment[char] for char in random_assignment]
        random_assign_prompt_string = "".join(random_assignment_prompt)

        text_sample = random_assign_prompt_string + " " + random_input_string + "=" + encoded_input_string
        tokenized_sample = []
        for char in text_sample:
            tokenized_sample += self.tokenizer.encode(char)
        tokenized_sample += [self.eos_token_id]
        
        prompt_length = len(random_assign_prompt_string) + len(random_input_string) + 2
        assignment_length = len(random_assign_prompt_string)
        input_string_length = len(random_input_string)

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
            attn_labels = self.generate_attention_labels(text_sample, input_ids, assignment_length, input_string_length,  prompt_length, preprompt_length, self.ignore_index)
        
        return text_sample, input_ids, target_ids, attn_labels
    


        """

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

        """
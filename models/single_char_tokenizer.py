from typing import List

vocabulary = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "-", "=", "+", "\n", "<END>", "<PAD>"]

class SingleCharTokenizer:
    def __init__(self, characters: List[str]):
        self.allocated_token_ids = len(characters) - 1
        self.char2token = {}
        self.token2char = {}
        for token, char in enumerate(characters):
            self.char2token[char] = token
            self.token2char[token] = char
    
    def tokenize(self, text: str) -> List[int]:
        tokenized = []
        for char in text:
            tokenized.append(self.char2token[char])
        return tokenized
    
    def encode(self, text: str) -> List[int]:
        return self.tokenize(text)
    
    def decode(self, tokens: List[int]) -> str:
        text = ""
        for token in tokens:
            if self.token2char.get(token) is None:
                text += "<UNK>"
            else:
                text += self.token2char[token]
        return text
    
    
        
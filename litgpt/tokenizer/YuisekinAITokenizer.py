# 参考
import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from transformers.utils.hub import cached_file


class YuisekinAITokenizer(PreTrainedTokenizer):
    def __init__(self, model_path="./tokenizer.json", pad="<PAD>", bos="<BOS>", eos="<EOS>", unk="<UNK>", mask="<MASK>", **kwargs):
        from tokenizers import Tokenizer

        try:
            self._tokenizer = Tokenizer.from_file(model_path)
        except Exception as e:
            print("exception: ", e)
            print("load from cache...")
            model_path = cached_file("yuiseki/sentencepiece_ja", "tokenizer.json")
            self._tokenizer = Tokenizer.from_file(model_path)
        super().__init__(**kwargs)
        self.add_special_tokens({"pad_token": pad, "bos_token": bos, "eos_token": eos, "unk_token": unk, "mask_token": mask})
        self._tokenizer.add_tokens([" ", "　"])

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def _tokenize(self, text, **kwargs):
        return self._tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token):
        ids = self._tokenizer.encode(token).ids
        if len(ids) == 0:
            return self.unk_token_id
        return self._tokenizer.encode(token).ids[0]

    def _convert_id_to_token(self, index: int) -> str:
        return self._tokenizer.decode([index])

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 日本語用
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.get_vocab().items(), key=lambda kv: kv[1]):
                if index != token_index:
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

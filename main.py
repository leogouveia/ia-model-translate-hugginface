from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence


class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}"
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [
            self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens
        ]


marian_pt_en = Translator("en", "ROMANCE")
text = marian_pt_en.translate([">>pt_br<< How are you, dear?"])
print(text)

marian_en_pt = Translator("ROMANCE", "en")
text = marian_en_pt.translate(["Tom esteve no havaí várias vezes."])
print(text)

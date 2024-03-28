# 参考:
#  - https://zenn.dev/if001/articles/6c507e15cd958b#training

import os
import time
from pathlib import Path

from litgpt import Tokenizer
from litgpt.data.prepare_starcoder import DataChunkRecipe
from litdata.processing.data_processor import DataProcessor

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset

import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

dataset_list = [
    {"id": "CohereForAI/aya_dataset", "config": None, "filter": {"field": "language_code", "value": "eng"}},
    {"id": "CohereForAI/aya_dataset", "config": None, "filter": {"field": "language_code", "value": "jpn"}},
    {"id": "wikimedia/wikipedia", "config": "20231101.en"},
    {"id": "wikimedia/wikipedia", "config": "20231101.ja"},
]


class YuisekinAIDataRecipe(DataChunkRecipe):
    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer
        self.total_token_cnt = 0

    def prepare_structure(self, input_dir):
        return dataset_list

    def prepare_item(self, dataset_data):
        print("start...", dataset_data)
        dataset_id = dataset_data["id"]
        dataset_config = dataset_data["config"]
        if dataset_config is not None:
            raw_dataset = load_dataset(dataset_id, dataset_config, split="train")
        else:
            raw_dataset = load_dataset(dataset_id, split="train")
        if "filter" in dataset_data:
            data_df = raw_dataset.to_pandas()
            filter_field = dataset_data["filter"]["field"]
            filter_value = dataset_data["filter"]["value"]
            data_df = data_df[data_df[filter_field] == filter_value]
            dataset = Dataset.from_pandas(data_df)
            ds = dataset
        else:
            ds = raw_dataset
        print("ds", ds)
        if "aya" in dataset_id:
            for v in ds["inputs"]:
                text_ids = self.tokenizer.encode(v, bos=False, eos=True)
                self.total_token_cnt += len(text_ids)
                yield text_ids
        else:
            for v in ds:
                text_ids = self.tokenizer.encode(v["text"], bos=False, eos=True)
                self.total_token_cnt += len(text_ids)
                yield text_ids


def prepare_for_dataset(
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = YuisekinAIDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=None,
        output_dir=str(destination_path),
        fast_dev_run=True,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


def prepare(
    destination_path: Path = Path("/data/pretrain_data/meta-llama/llama-2-7b-hf/"),
    # 2048 block size + 1 for causal (from LLama), 1024 blocks
    chunk_size: int = 2049 * 1024,
) -> None:
    tokenizer_path = Path("checkpoints/meta-llama/Llama-2-7b-hf")
    prepare_for_dataset(
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    prepare()

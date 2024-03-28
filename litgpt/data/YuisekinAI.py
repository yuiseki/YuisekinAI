from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule


@dataclass
class YuisekinAI(DataModule):
    data_path: Union[str, Path] = Path("/data/pretrain_data/meta-llama/llama-2-7b-hf")
    seed: int = 42
    num_workers: int = 8

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1

    def prepare_data(self) -> None:
        for path in [self.data_path]:
            if not Path(path).is_dir():
                raise FileNotFoundError("YuisekinAIの学習データが見つかりませんでした。データを準備して、data_pathに指定してください。")

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=self.data_path,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )

        # Mix SlimPajama data and Starcoder data with these proportions:
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

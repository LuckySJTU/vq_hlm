import torch
import datasets
from datasets import Dataset, DatasetInfo, BuilderConfig, GeneratorBasedBuilder
import os

class MyDataset(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            description="自定义数据集",
            # features={
            #     "input_ids": torch.dtypes.LongTensor,
            #     "label": datasets.Value('int32')
            # },
            supervised_keys=("input_ids", "label"),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "train_tokens.pt"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": "val_tokens.pt"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": "test_tokens.pt"}),
        ]

    def _generate_examples(self, filepath):
        raw_token = torch.load(os.path.join(self.base_path, filepath), weights_only=True)
        for i in range(0, len(raw_token)-1, 1024):
            input_ids = raw_token[i:i+1024]
            label = raw_token[i+1:i+1025]
            if len(input_ids) > len(label):
                input_ids = input_ids[:len(label)]
            if len(input_ids) < 1024:
                input_ids = torch.cat((input_ids, torch.zeros((1024-input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype)))
                label = torch.cat((label, torch.zeros((1024-label.shape[0], label.shape[1]), dtype=label.dtype)*(-100)))
                # input_ids = torch.cat((input_ids, raw_token[:1024-input_ids.shape[0]]))
                # label = torch.cat((label, raw_token[1:1025-label.shape[0]]))
            yield i, {"input_ids": input_ids, "label": label}

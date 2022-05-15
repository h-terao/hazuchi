from torch.utils import data
import numpy as np

from hazuchi.torch_utils import collate_fun

# import torch.multiprocessing as multiprocessing

# multiprocessing.set_start_method("spawn")


class NumpyDataSet(data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.images = np.random.rand(num_samples, 32, 32, 3)
        self.labels = np.random.randint(0, 100, (self.num_samples,))

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return {
            "image": [image, image],
            "label": {
                "label": label,
                "label2": label,
            },
        }

    def __len__(self):
        return self.num_samples


def main():
    loader = data.DataLoader(
        NumpyDataSet(1000),
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fun,
        drop_last=True,
        num_workers=4,
    )

    for batch in loader:
        print("x")


if __name__ == "__main__":
    main()

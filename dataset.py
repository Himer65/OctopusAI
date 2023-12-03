import torch
from torch.utils.data import Dataset


class Flickr8k(Dataset):
    def __init__(self,
                 path: str,
                 device: str = "cpu"):
        data = torch.load(path)
        self.x = data["x"].to(device)
        self.y = data["y"].to(device)[:, :, 0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, key: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[key] / 127.5 - 1
        y = self.y[:, key]

        return (x, y)


if __name__ != "__main__":
    dataset = Flickr8k(
        path = "/content/drive/MyDrive/data/flickr8k/flickr8k.bin",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )

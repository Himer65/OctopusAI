import torch
from torch.utils.data import Dataset


class Flickr8k(Dataset):
    def __init__(self,
                 path: str,
                 device: str = "cpu"):
        data = torch.load(path, map_location=device)
        self.x = data["x"]
        self.y = data["y"]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, key: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[key] / 127.5 - 1
        y = self.y[key]

        return (x, y)

if __name__ != "__main__":
    dataset = Flickr8k(
        path = "/content/drive/MyDrive/data/flickr8k/flickr8k.bin",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
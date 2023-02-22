# import torch
# from multi_head import MultiHead
# from multi_loss import MultiLoss

# CHAR_NUM = 10

# x = torch.rand(2, 512, 1, 40)
# targets = [
#     torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.int),
#     torch.tensor([[10, 1, 2, 3, 4, 5, 6, 7, 8, 10], [10, 1, 2, 3, 4, 5, 6, 7, 8, 10]], dtype=torch.int),
#     torch.tensor([10, 10], dtype=torch.float32)
# ]
# model = MultiHead(in_channels=512, char_num=CHAR_NUM, max_text_length=25)
# model.train()

# print(model)
# output = model(x, targets)

# if model.training:
#     ctc_head_out, sar_head_out = output

#     loss_fn = MultiLoss()
#     loss = loss_fn(ctc_head_out, targets[0], sar_head_out, targets[1], length=targets[2])
#     print(loss)

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import json
import os
import pytorch_lightning as pl
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from multi_head_sar_svtr_model import MultiHeadSARSVTRModel
from multi_loss import MultiLoss
import time
from torchvision.models import regnet_y_800mf


class RecognitionDataset(Dataset):

    def __init__(self, max_text_length=25):
        self.max_text_length = max_text_length
        self.characters = list("0123456789abcdefghijklmnopqrstuvwxyz$%&*()_-+=#@!/,.?~")

        self.crops = list()
        self.words = list()
        for _, _, files in os.walk("../datasets/CORD/train/image/"):
            for f in files:
                try:
                    file_idx = f.replace("receipt_", "").replace(".png", "")

                    with open(f"../datasets/CORD/train/json/receipt_{file_idx}.json", "rb") as f:
                        annotations = json.load(f)

                    img = cv2.imread(f"../datasets/CORD/train/image/receipt_{file_idx}.png", cv2.IMREAD_COLOR)
                    if img is False or img is None:
                        continue
                except FileNotFoundError:
                    continue

                h, w, _ = img.shape
                img_mask = np.zeros((h, w, 1), dtype=np.uint8)
                for line in annotations["valid_line"]:
                    for word in line["words"]:
                        text = word["text"].lower()

                        # Remove small or big words
                        if len(text) < 3 or len(text) > (max_text_length - 2):
                            continue

                        # Remove unknown characters
                        unk_characters = list(filter(lambda x: x not in self.characters, text))
                        if len(unk_characters) > 0:
                            continue

                        points = word["quad"]

                        min_x, min_y, max_x, max_y = [points["x1"], points["y1"], points["x2"], points["y4"]]

                        points = [
                            [points["x1"], points["y1"]],
                            [points["x2"], points["y2"]],
                            [points["x3"], points["y3"]],
                            [points["x4"], points["y4"]],
                        ]

                        # Fill mask where the text is located
                        cv2.fillPoly(img_mask, np.array([points], dtype=np.int32), 255)

                        # Merge mask and original images
                        text_image = cv2.bitwise_and(img.copy(), img.copy(), mask=img_mask)

                        # Crop text area
                        crop = text_image[min_y:max_y, min_x:max_x]

                        # Set text crop to top-left of a new image
                        if crop.shape[0] > 48:
                            crop = cv2.resize(crop, (crop.shape[1], 48))

                        if crop.shape[1] > 320:
                            crop = cv2.resize(crop, (320, crop.shape[0]))

                        text_crop = np.zeros([48, 320, 3], dtype=np.uint8)
                        text_crop[0:crop.shape[0], 0:crop.shape[1]] = crop

                        # Append data
                        self.crops.append(text_crop)
                        self.words.append(text)

                if len(self.crops) >= 100:
                    break

        self.transforms = T.Compose(
            [T.ToTensor(),
             T.ConvertImageDtype(torch.float),
             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return len(self.words)

    def get_characters(self):
        return self.characters

    def CTC_label_encode(self, label):
        ctc_characters = ['BLANK'] + self.characters
        encoded_label = list()
        for char in label:
            encoded_label.append(ctc_characters.index(char))

        # PAD label
        encoded_label = encoded_label + [0] * (self.max_text_length - len(encoded_label))

        return torch.tensor(encoded_label, dtype=torch.long)

    def SAR_label_encode(self, label):
        sar_characters = self.characters + ['BOS/EOS'] + ['PAD']
        pad_idx = len(sar_characters) - 2
        bos_eos_idx = len(sar_characters) - 2

        encoded_label = list()
        for char in label:
            encoded_label.append(sar_characters.index(char))

        encoded_label = [bos_eos_idx] + encoded_label + [bos_eos_idx]

        # PAD label
        encoded_label = encoded_label + [pad_idx] * (self.max_text_length - len(encoded_label))

        return torch.tensor(encoded_label, dtype=torch.long)

    def __getitem__(self, index):
        image = self.crops[index]
        text = self.words[index]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert('RGB')

        image_tensor = self.transforms(image)
        ctc_label = self.CTC_label_encode(text)
        sar_label = self.SAR_label_encode(text)

        return image_tensor, ctc_label, sar_label, torch.tensor(len(text), dtype=torch.int)


class MultiHeadRecognizer(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = MultiHeadSARSVTRModel(**kwargs)
        self.loss_fn = MultiLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images_tensor, ctc_label, sar_label, length = batch

        # DEBUG
        # images_tensor = torch.rand(images_tensor.shape[0], 512, 1, 40).float().cuda()

        ctc_head_out, sar_head_out = self.model(images_tensor, sar_label)

        loss = self.loss_fn(
            ctc_head_out,
            ctc_label,
            length,
            sar_head_out,
            sar_label,
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images_tensor, ctc_label, _, length = batch

        ctc_head_out = self.model(images_tensor)

        loss = self.loss_fn(ctc_head_out, ctc_label, length=length)
        self.log("val_loss", loss)

        return loss


if __name__ == "__main__":
    dataset = RecognitionDataset(max_text_length=25)
    dataloader = DataLoader(dataset, batch_size=50, num_workers=1, shuffle=True)

    model = MultiHeadRecognizer(in_channels=512, n_characters=len(dataset.get_characters()), max_text_length=25)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1)

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="gpu",
        devices=1,
        # check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        # callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dataloader)

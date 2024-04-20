from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, args,transforms=None):
        super(MyDataset, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(args.mae_path)
        
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) >= num_class:
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        image = self.processor(img).pixel_values
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)
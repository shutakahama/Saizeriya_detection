from glob import glob
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torchvision import transforms


def xml2list(xml_path, classes):
    xml = ET.parse(xml_path).getroot()
    boxes, labels = [], []

    for obj in xml.iter('object'):
        label = obj.find('name').text

        if label in classes:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text.split(".")[0])
            ymin = int(bndbox.find('ymin').text.split(".")[0])
            xmax = int(bndbox.find('xmax').text.split(".")[0])
            ymax = int(bndbox.find('ymax').text.split(".")[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(classes.index(label))

    anno = {'bboxes': boxes, 'labels': labels}
    return anno, len(boxes)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, xml_paths, classes):
        super().__init__()
        self.image_dir = image_dir
        self.xml_paths = xml_paths
        self.image_ids = sorted(glob('{}/*'.format(xml_paths)))
        self.classes = classes
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image_id = self.image_ids[index].split("/")[-1].split(".")[0]
        image = Image.open(f"{self.image_dir}/{image_id}.png")
        image = self.transform(image)
        image = image[:3, :, :]

        path_xml = f'{self.xml_paths}/{image_id}.xml'
        annotations, obje_num = xml2list(path_xml, self.classes)

        boxes = torch.as_tensor(annotations['bboxes'], dtype=torch.int64)
        labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)

        # boxes = boxes * ratio
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((obje_num,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels + 1
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target, image_id

    def __len__(self):
        return len(self.image_ids)

from glob import glob
import argparse
import cv2
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import TensorDataset
from dataloader import MyDataset


def FasterRCNN(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    xml_dir = args.base_path + "data/label"
    img_dir = args.base_path + "data/img"
    test_dir = args.base_path + "data/test"
    save_path = args.base_path + "result"

    dataset_class = ['Item']
    colors = ((0, 0, 0), (255, 0, 0))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = MyDataset(img_dir, xml_dir, dataset_class)
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)

    model = FasterRCNN(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_dataloader):
            images, targets, image_ids = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item() * len(images)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(i, train_loss)
            if i >= 0:
                break

        print(f"epoch {epoch+1} loss: {train_loss / len(train_dataloader.dataset)}")
        torch.save(model, 'model.pt')
        test(model, dataset_class, colors, test_dir, save_path, device)


def test(model, dataset_class, colors, test_dir, save_path, device):
    model.eval()
    test_classes = ['__background__'] + dataset_class
    for imgfile in sorted(glob(test_dir + '/*')):
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = torchvision.transforms.functional.to_tensor(img)

        with torch.no_grad():
            prediction = model([image_tensor.to(device)])

        for i, box in enumerate(prediction[0]['boxes'][:10]):
            score = prediction[0]['scores'][i].cpu().numpy()
            score = round(float(score), 2)
            cat = prediction[0]['labels'][i].cpu().numpy()

            txt = '{} {}'.format(test_classes[int(cat)], str(score))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            c = colors[int(cat)]
            box = box.cpu().numpy().astype('int')
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), c, 2)
            cv2.rectangle(img, (box[0], box[1] - cat_size[1] - 2), (box[0] + cat_size[0], box[1] - 2), c, -1)
            cv2.putText(img, txt, (box[0], box[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        file_name = imgfile.split("/")[-1].split(".")[0]
        pil_img = Image.fromarray(img)
        pil_img.save(save_path + f"/{file_name}_result.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='machigai-sagashi')
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--base_path", type=str, default='./')
    args = parser.parse_args()

    train()

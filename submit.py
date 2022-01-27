import json
import torch
import torchvision.transforms as T

from utils.augment import TrivialAugmentWide
from utils.dataset import UnlabeledDataset
from models import create_resnet18


data_path = './data/test/images'
state_dict_path = './weights/model_2.pth'
output_file = 'sub.csv'
NUM_TRY = 1

augment = T.Compose([
    TrivialAugmentWide(),
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = UnlabeledDataset(data_path, transforms=augment)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

model = create_resnet18(1, pretrained=False)
model.load_state_dict(torch.load(state_dict_path))
model = model.eval().cuda()

ids = [int(x[1][:x[1].find('.')]) for x in dataset.im_list]
grand_results = []
for _ in range(NUM_TRY):
    results = []
    with torch.no_grad():
        for imgs in loader:
            with torch.cuda.amp.autocast():
                preds = torch.sigmoid(model(imgs.cuda())).squeeze()
            results.append(preds.cpu())
    grand_results.append(torch.cat(results))

final = torch.zeros(grand_results[0].shape)
for i in range(len(grand_results)):
    final += grand_results[i]
final /= NUM_TRY
final = final.cpu().tolist()

results = list(zip(final, ids))

rs = sorted(results, key=lambda x: x[0], reverse=True)

# EZ trick EZ *language model*
meta = json.load(open('./data/test/meta.json'))["images"]

no_flood = []
for im_info in meta:
    desc = im_info["description"]
    if desc is None: desc = ""
    tags = im_info["user_tags"]
    if tags is None: tags = [""]
    if len(tags) == 0: tags = [""]
    tagstr = ""
    for tag in tags: tagstr += tag + " "
    title = im_info["title"]
    if title is None: title = ""
    lol = title + " " + tagstr + " " + desc
    if 'flood' not in lol:
        no_flood.append(int(im_info["image_id"]))


final_sub = []
see_u_later = []
for idx, pred in enumerate(rs):
    if pred[1] in no_flood:
        print(pred[1])
        print(f"At {idx}")
        see_u_later.append(pred[1])
    else:
        final_sub.append(pred[1])


final_sub += see_u_later

with open("sub.csv", "w+") as f:
    f.write("Id\n")
    for i in final_sub:
        f.write(f"{i}\n")
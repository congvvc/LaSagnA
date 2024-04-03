import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor


from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .util.coco_map import COCO_ID_MAP, COCO_CATEGORIES

from .utils import SHORT_QUESTION_LIST, SEG_INPUT_LIST, SEG_ANSWER_LIST, SEG_INS_OUTPUT, SEG_NO_INS_OUTPUT

CITYSCAPES_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

ADE20K_classes = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk",
    "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa",
    "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
    "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow",
    "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal",
    "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag"
]



def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary_vistas")
    with open(os.path.join(mapillary_data_root, "validation/panoptic/panoptic_2018.json")) as f:
        mapillary_classes = json.load(f)["categories"]
        mapillary_classes = [x['name'].lower() for x in mapillary_classes]

    # mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels

def init_cityscapes(base_image_dir):
    cityscapes_data_root = os.path.join(base_image_dir, "cityscapes/gtFine")
    with open(os.path.join(cityscapes_data_root, "cityscapes_panoptic_val.json")) as f:
        cityscapes_classes = json.load(f)["categories"]
        cityscapes_classes = {x['name'].lower(): x['id'] for x in cityscapes_classes}

    # cityscapes_classes = np.array(cityscapes_classes)
    cityscapes_labels = []
    for kk in os.listdir(os.path.join(cityscapes_data_root, "train")):
        cur_cityscapes_labels = sorted(
            glob.glob(
                os.path.join(cityscapes_data_root, "train", kk, "*labelIds.png")))
        cityscapes_labels.extend(cur_cityscapes_labels)

    cityscapes_images = [
        x.replace('gtFine_labelIds', 'leftImg8bit').replace('gtFine', 'leftImg8bit')
        for x in cityscapes_labels]

    print("cityscapes: ", len(cityscapes_images))
    return cityscapes_classes, cityscapes_images, cityscapes_labels


def init_ade20k(base_image_dir):
    # with open("utils/ade20k_classes.json", "r") as f:
    #     ade20k_classes = json.load(f)
    ade20k_classes = ADE20K_classes
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ADEChallengeData2016/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ADEChallengeData2016",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_coco_sem(base_image_dir):
    cocostuff_classes = []
    
    cocostuff_classes = [_["name"].lower().replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in COCO_CATEGORIES]    
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    # cocostuff_labels = glob.glob(
    #     os.path.join(base_image_dir, "coco", "stuffthingmaps_trainval2017",'train2017', "*.png")
    # )
    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "coco", "panoptic_semseg_train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("panoptic_semseg_train2017", "train2017") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []

    with open("./dataset/coco/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])

    cocostuff_classes = [_.lower().replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in cocostuff_classes]

    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "coco", "stuffthingmaps_trainval2017",'train2017', "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("/stuffthingmaps_trainval2017", "") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_openimage(data_split = 'test'):

    if data_split == 'train':
        image_path = './dataset/OpenImageV6/folder_train'
    
    else:
        image_path = f'./dataset/OpenImageV6/{data_split}'
    mask_path = f'./dataset/OpenImageV6/{data_split}-masks'
    mask_anno_path = f'./dataset/OpenImageV6/annotations/new/{data_split}_mask.json'

    with open(mask_anno_path, 'r') as f:
        mask_anno = json.load(f)

    openimage_images = []
    openimage_labels = []
    for img, mask_dic in mask_anno.items():
        
        if img != 'categories':
            openimage_images.append(os.path.join(image_path, img[:2], img+'.jpg'))
            # mask_dic: {"suit": ["d/d0ed76e0533a914d_m01xyhv_cffd8afa.png"], "tie": ["d/d0ed76e0533a914d_m01rkbr_5f6546a2.png"], "shirt": ["d/d0ed76e0533a914d_m01n4qj_134cdeb0.png"]}
            openimage_labels.append(mask_dic)
            # for mask_class, mask_list in mask_dic.items():
            #     mask_path_list = [os.path.join(mask_path, kk) for kk in mask_list]
        else:
            openimage_classes = mask_dic

            
    # openimage_classes = [_.replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in openimage_classes]    
    openimage_classes = np.array(openimage_classes)
    
    print("openimage: ", len(openimage_images))
    return openimage_classes, openimage_images, openimage_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "paco_lvis_v1", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


def get_sample_classes(all_classes, unique_label, num_all_classes, ds):
    
    coco_171_classes = [_["name"].lower().replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in COCO_CATEGORIES]
    
    if ds == 'cityscapes':
        all_positive_classes = []
        for class_id in unique_label:
            for c, i in all_classes.items():
                if i == class_id:
                    all_positive_classes.append(c)
        all_classes = [c for c,i in all_classes.items()]
        
    elif ds == 'openimage':
        all_positive_classes = unique_label
    else:
        all_positive_classes = [all_classes[class_id] for class_id in unique_label]

    if ds == 'cocostuff':
        all_classes = coco_171_classes

    all_negtive_classes = [ii for ii in all_classes if ii not in all_positive_classes]
    
    if len(all_classes) < num_all_classes:
        sampled_pos_classes = all_positive_classes
        sampled_neg_classes = all_negtive_classes
    # maybe not happen
    elif len(all_positive_classes) >= num_all_classes:
        sampled_pos_classes = np.random.choice(
            all_positive_classes, size=40, replace=False
        ).tolist()
        sampled_neg_classes = np.random.choice(
            all_negtive_classes, size=20, replace=False
        ).tolist()
    else:
        sampled_pos_classes = all_positive_classes
        sampled_neg_classes = np.random.choice(
            all_negtive_classes, size=num_all_classes-len(sampled_pos_classes), replace=False
        ).tolist()


    sampled_classes = sampled_pos_classes + sampled_neg_classes
    random.shuffle(sampled_classes)
    return all_classes, sampled_classes, sampled_pos_classes, sampled_neg_classes

class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        num_all_classes: int = 100,
        exclude_val=False,
        sem_seg_data="cocostuff||ade20k||mapillary||cityscapes||pascal_part||paco_lvis",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.num_all_classes = num_all_classes

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.seg_input_list = SEG_INPUT_LIST
        self.seg_answer_list = SEG_ANSWER_LIST

        self.data_split = 'train'


        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            if ds == 'openimage':
                classes, images, labels = eval("init_{}".format(ds))(self.data_split)
            else:
                classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        # if "cocostuff" in self.sem_seg_datas:
        #     self.cocostuff_class2index = {
        #         c: i for i, c in enumerate(self.data2classes["cocostuff"])
        #     }

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # sample from "cocostuff||ade20k||mapillary||cityscapes||openimage||pascal_part||paco_lvis"
        sample_rate = np.array([6, 3, 2, 2, 1])  # 6, 3, 2, 2, 1
        sample_rate = sample_rate / sample_rate.sum()
        ind = np.random.choice(list(range(len(self.sem_seg_datas))), p=sample_rate)
        # ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ind]
        use_all_neg = False

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            elif 0< len(anns) <= 30:
                sampled_anns = anns

            elif len(anns) > 30:
                sampled_anns = np.random.choice(anns, size=30, replace=False).tolist()
            
            # sampled_anns = anns
            random.shuffle(sampled_anns)
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)
            cur_positive_num = len(sampled_classes)
            sampled_pos_classes = sampled_classes

        elif ds in ["ade20k", "cocostuff", "mapillary", "cityscapes"]:

            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cityscapes":
                label[label<7] = 255
            # elif ds == "cocostuff":
            #     for c, i in self.cocostuff_class2index.items():
            #         if "-" in c:
            #             label[label == i] = 255
            elif ds == "mapillary":
                label[label == 65] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = image.shape[:2]
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)

            if ds == "cocostuff":
                new_unique_label = []
                for uni_label in unique_label:
                    if uni_label in COCO_ID_MAP:
                        new_unique_label.append(uni_label)
                    else:
                        print('not in 171 cocostuff classes')
                unique_label = new_unique_label
            

            if len(unique_label) == 0:
                return self.__getitem__(0)
            
            # copy before random.shuffle
            dataclasses = self.data2classes[ds].copy()
            
            # random_neg_p = np.random.rand()
            # if random_neg_p < 0.2:
            #     use_all_neg = True

            all_classes, sampled_classes, sampled_pos_classes, sampled_neg_classes = get_sample_classes(dataclasses, unique_label, self.num_all_classes, ds)


        elif ds in ["openimage"]:
            mask_path = f'./datasets/OpenImageV6/{self.data_split}-masks'
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = image.shape[:2]
            unique_label = []
            # mask_dic: {"suit": ["d/d0ed76e0533a914d_m01xyhv_cffd8afa.png"], "tie": ["d/d0ed76e0533a914d_m01rkbr_5f6546a2.png"], "shirt": ["d/d0ed76e0533a914d_m01n4qj_134cdeb0.png"]}
            mask_dic = labels[idx]
            masks = []
            for mask_class, mask_list in mask_dic.items():
                unique_label.append(mask_class)
                # multi mask for cur class, integrate them
                if len(mask_list) > 1:
                    cur_class_mask = torch.zeros(ori_size).bool()
                    for kk in mask_list:
                        cur_mask = Image.open(os.path.join(mask_path, kk))
                        cur_mask = torch.from_numpy(np.array(cur_mask))
                        if cur_mask.shape[0] != image.shape[0] or cur_mask.shape[1] != image.shape[1]:
                            cur_mask = cur_mask.float().unsqueeze(0).unsqueeze(0)
                            cur_mask = F.interpolate(cur_mask, size=ori_size, mode='nearest')
                            cur_mask = cur_mask[0,0].bool()
                        cur_class_mask = cur_class_mask | cur_mask
                    masks.append(cur_class_mask)
                else:
                    # only one mask for cur class
                    cur_mask = Image.open(os.path.join(mask_path, mask_list[0]))
                    cur_mask = torch.from_numpy(np.array(cur_mask))
                    if cur_mask.shape[0] != image.shape[0] or cur_mask.shape[1] != image.shape[1]:
                        cur_mask = cur_mask.float().unsqueeze(0).unsqueeze(0)
                        cur_mask = F.interpolate(cur_mask, size=ori_size, mode='nearest')
                        cur_mask = cur_mask[0,0].bool()

                    masks.append(cur_mask)
                    
            label = torch.ones(ori_size) * self.ignore_label
            masks = torch.stack(masks, dim=0)
            
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]

            # copy before random.shuffle
            dataclasses = self.data2classes[ds].copy()
            
            all_classes, sampled_classes, sampled_pos_classes, sampled_neg_classes = get_sample_classes(dataclasses, unique_label, self.num_all_classes, ds)


        questions = []
        answers = []
        class_ids = []
        if ds == 'openimage':
            sampled_classes_question = sampled_classes.copy()
            random.shuffle(sampled_classes_question) # question random differ from answer
            this_sampled_classes = ', '.join(sampled_classes_question)
            this_seg_input = random.choice(self.seg_input_list)
            questions.append(this_seg_input.format(all_class=this_sampled_classes.lower()))
        else:
            random.shuffle(all_classes)
            this_sampled_classes = ', '.join(all_classes)
            this_seg_input = random.choice(self.seg_input_list)
            questions.append(this_seg_input.format(all_class=this_sampled_classes.lower()))
            
        cur_answer = ''
        # cur_seg_answer = random.choice(self.seg_answer_list)
        # cur_answer += cur_seg_answer

        for sampled_cls_id in range(len(sampled_classes)):
            text = sampled_classes[sampled_cls_id]
            if text in sampled_pos_classes:
                cur_answer += SEG_INS_OUTPUT.format(class_name=text.lower())
            elif text in sampled_neg_classes:
                cur_answer += SEG_NO_INS_OUTPUT.format(class_name=text.lower())

            assert len(text.split("||")) == 1

            if ds in ["paco_lvis", "pascal_part", "openimage"]:
                continue
            
            if text in sampled_pos_classes:
                if ds == 'cityscapes':
                    class_id = self.data2classes[ds][text]
                else:
                    class_id = self.data2classes[ds].tolist().index(text)
                class_ids.append(class_id)

        answers.append(cur_answer)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        elif ds != 'openimage':
            if use_all_neg:
                masks = torch.rand(0, *ori_size)
                label = torch.ones(ori_size) * self.ignore_label
            else:
                label = torch.from_numpy(label).long()
                masks = []
                for class_id in class_ids:
                    masks.append(label == class_id)
                masks = torch.stack(masks, dim=0)
    
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_pos_classes,
            ds,
        )

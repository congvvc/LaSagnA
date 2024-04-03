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
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import (IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import (SEG_INPUT_LIST, SEG_NO_INS_OUTPUT,DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN,
                    INS_TOKEN)

from .util.coco_map import COCO_ID_MAP, COCO_CATEGORIES

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
            os.path.join(mapillary_data_root, "validation", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    # with open("utils/ade20k_classes.json", "r") as f:
    #     ade20k_classes = json.load(f)
    ade20k_classes = ADE20K_classes
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ADEChallengeData2016/images", "validation"))
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
                "validation",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels



def init_cityscapes(base_image_dir):
    cityscapes_data_root = os.path.join(base_image_dir, "cityscapes/gtFine")
    with open(os.path.join(cityscapes_data_root, "cityscapes_panoptic_val.json")) as f:
        cityscapes_classes_dic = json.load(f)["categories"]
        cityscapes_classes = {x['name'].lower(): x['id'] for x in cityscapes_classes_dic}
        cityscapes_classes_continue = {x['name'].lower(): kk for kk, x in enumerate(cityscapes_classes_dic)}

    # cityscapes_classes = np.array(cityscapes_classes)
    cityscapes_labels = []
    for kk in os.listdir(os.path.join(cityscapes_data_root, "val")):
        cur_cityscapes_labels = sorted(
            glob.glob(
                os.path.join(cityscapes_data_root, "val", kk, "*labelIds.png")))
        cityscapes_labels.extend(cur_cityscapes_labels)

    cityscapes_images = [
        x.replace('gtFine_labelIds', 'leftImg8bit').replace('gtFine', 'leftImg8bit')
        for x in cityscapes_labels]

    print("cityscapes: ", len(cityscapes_images))
    return cityscapes_classes, cityscapes_classes_continue, cityscapes_images, cityscapes_labels




def init_coco_sem(base_image_dir):
    cocostuff_classes = []
    
    cocostuff_classes = [_["name"].lower().replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in COCO_CATEGORIES]    
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    # cocostuff_labels = glob.glob(
    #     os.path.join(base_image_dir, "coco", "stuffthingmaps_trainval2017",'train2017', "*.png")
    # )
    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "coco", "panoptic_semseg_val2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("panoptic_semseg_val2017", "val2017") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels



def init_cocostuff(base_image_dir):
    cocostuff_classes = []

    with open(".dataset/coco/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])

    cocostuff_classes = [_.lower().replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in cocostuff_classes]

    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "coco", "stuffthingmaps_trainval2017",'val2017', "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("/stuffthingmaps_trainval2017", "") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels





def get_sample_classes(all_classes, unique_label, num_all_classes,  ds):
    
    coco_171_classes = [_["name"].lower().replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in COCO_CATEGORIES]

    if ds == 'cityscapes':
        all_positive_classes = []
        for class_id in unique_label:
            for c, i in all_classes.items():
                if i == class_id:
                    all_positive_classes.append(c)
        all_classes = [c for c,i in all_classes.items()]
    else:
        all_positive_classes = [all_classes[class_id] for class_id in unique_label]
        all_classes = all_classes.tolist()
    # random.shuffle(all_positive_classes)
    # random.shuffle(sampled_classes)
        
    if ds == 'cocostuff':
        all_classes = coco_171_classes
    
    sampled_classes = all_classes
    
    
    return sampled_classes, len(all_positive_classes), all_positive_classes


class TestSemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        image_size: int = 1024,
        num_all_classes: int = 100,
        test_class_internal: int = 2,
        sem_seg_data="ade20k",
        use_all_classes=False,
        random_test=False,
    ):
        self.test_class_internal = test_class_internal
        self.sem_seg_data = sem_seg_data
        self.num_all_classes = num_all_classes
        self.use_all_classes = use_all_classes
        self.random_test = random_test

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)


        if sem_seg_data == 'cityscapes':
            self.classes, self.classes_continue, self.images, self.labels = eval("init_{}".format(sem_seg_data))(base_image_dir)
        else:
            self.classes, self.images, self.labels = eval("init_{}".format(sem_seg_data))(base_image_dir)

        # if sem_seg_data == "cocostuff":
        #     self.cocostuff_class2index = {
        #         c: i for i, c in enumerate(self.classes)
        #     }

    def __len__(self):
        return len(self.images)

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
        ds = self.sem_seg_data

        if ds in ["ade20k", "cocostuff", 'cityscapes', "mapillary"]:

            image, labels = self.images, self.labels
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
            elif ds == "mapillary":
                label[label >= 65] = 255
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
            if len(unique_label) == 0:
                return self.__getitem__(0)


            sampled_classes, cur_positive_num, sampled_pos_classes = get_sample_classes(self.classes, unique_label, self.num_all_classes, ds)




        questions = []
        answers = []
        class_ids = []
        class_ids_continue = []
        this_sampled_classes = ', '.join(sampled_classes)
        questions.append(SEG_INPUT_LIST[0].format(all_class=this_sampled_classes.lower()))
        cur_answer = ''
        answers.append(cur_answer)

        # kk = 0
        # test_class_internal = self.test_class_internal
        # while kk < len(sampled_classes):
        #     kk_up = min(kk + test_class_internal, len(sampled_classes))
        #     cur_sampled_classes = sampled_classes[kk:kk_up]
        #     cur_sampled_classes = ' '.join(cur_sampled_classes)
        #     questions.append(SEG_INPUT.format(all_class=cur_sampled_classes.lower()))
        #     cur_answer = ''
        #     answers.append(cur_answer)
        #     kk += test_class_internal


        for sampled_cls_id in range(cur_positive_num):
            text = sampled_pos_classes[sampled_cls_id]
            # cur_answer += SEG_OUTPUT.format(class_name=text.lower())

            assert len(text.split("||")) == 1

            # if ds in ["paco_lvis", "pascal_part"]:
            #     continue
            
            if ds == 'cityscapes':
                class_id = self.classes[text]
                class_id_continue = self.classes_continue[text]
                class_ids_continue.append(class_id_continue)
            elif ds == 'cocostuff':
                class_id = self.classes.tolist().index(text)
                class_id_ = sampled_classes.index(text)
                class_ids_continue.append(class_id_)
            else:
                class_id = self.classes.tolist().index(text)
            class_ids.append(class_id)


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

        label = torch.from_numpy(label).long()
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)

        if ds == 'cityscapes' or ds == 'cocostuff':
            out_class_ids = class_ids_continue
        else:
            out_class_ids = class_ids
        # used in class index during infer 
        if ds == 'cocostuff':
            sampled_classes = self.classes.tolist()

        inference = True
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            out_class_ids,
            sampled_pos_classes,
            sampled_classes,
            ds,
            inference,
        )




# 调整输入的batch中的输入
def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    sampled_classes_name_list = []
    sampled_all_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    data_name_list = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes_idx,
        sampled_classes_name,
        sampled_all_classes,
        data_name,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes_idx)
        sampled_classes_name_list.append(sampled_classes_name)
        sampled_all_classes_list.append(sampled_all_classes)
        len1 = len(conversations)
        # len2 = len(sampled_classes)
        # if masks.shape[0] == 0:
        #     off_len = len1
        # else:
        #     off_len = len2
        cnt += len1
        offset_list.append(cnt)
        inferences.append(inference)
        data_name_list.append(data_name)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    # if conv_type == "llava_v1":
    #     sep = conv.sep + conv.roles[1] + ": "
    # else:
    #     sep = "[/INST] "
    # for conversation, target in zip(conversation_list, targets):
    #     total_len = int(target.ne(tokenizer.pad_token_id).sum())

    #     rounds = conversation.split(conv.sep2)
    #     cur_len = 1
    #     target[:cur_len] = IGNORE_INDEX
    #     for i, rou in enumerate(rounds):
    #         if rou == "":
    #             break

    #         parts = rou.split(sep)
    #         # if len(parts) != 2:
    #         #     break
    #         assert len(parts) == 2, (len(parts), rou)
    #         parts[0] += sep

    #         if DEFAULT_IMAGE_TOKEN in conversation:
    #             round_len = len(tokenizer_image_token(rou, tokenizer))
    #             instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
    #         else:
    #             round_len = len(tokenizer(rou).input_ids)
    #             instruction_len = len(tokenizer(parts[0]).input_ids) - 2

    #         target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

    #         cur_len += round_len
    #     target[cur_len:] = IGNORE_INDEX


    #     if cur_len < tokenizer.model_max_length:
    #         assert cur_len == total_len



    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "classes_index": sampled_classes_list,
        "classes_name": sampled_classes_name_list,
        "all_classes": sampled_all_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "data_name_list": data_name_list,
    }


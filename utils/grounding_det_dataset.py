import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .box_ops import box_xyxy_to_cxcywh


from .utils import ANSWER_LIST_DET, SHORT_QUESTION_LIST_DET, LONG_QUESTION_LIST_DET, INS_TOKEN, DET_ANSWER_LIST


class GroundingDetDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        gr_det_data="genome||flickr",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST_DET
        self.answer_list = ANSWER_LIST_DET
        self.long_question_list = LONG_QUESTION_LIST_DET
        self.det_answer_list = DET_ANSWER_LIST

        self.ground_det_ds_list = gr_det_data.split(
            "||"
        )  # ['genome', 'flickr']
        self.ground_det_data = {}
        for ds in self.ground_det_ds_list:
            ground_det_ds = {}
            ground_det_ds["images"] = []
            ground_det_ds["ann"] = []
            if ds == "genome":
                ann_path = './dataset/rec_GC_genome196_train.json'
                with open(ann_path, 'r', encoding='utf8') as f:
                    for line in f:
                        cur_data = json.loads(line)
                        cur_img_path = cur_data['img_path'].replace('images2/','').replace('images/','')
                        cur_img_path = os.path.join('./dataset/VisualGenome', cur_img_path)
                        ground_det_ds["images"].append(cur_img_path)
                        ground_det_ds["ann"].append(cur_data['ann'])
            elif ds == "flickr":
                ann_path = './dataset/rec_CWB_flickr30k_train.json'
                with open(ann_path, 'r', encoding='utf8') as f:
                    for line in f:
                        cur_data = json.loads(line)
                        cur_img_path = cur_data['img_path'] + '.jpg'
                        cur_img_path = os.path.join('./dataset/flickr30k/flickr30k_images/train', cur_img_path)
                        ground_det_ds["images"].append(cur_img_path)
                        ground_det_ds["ann"].append(cur_data['ann'])
            
            self.ground_det_data[ds] = ground_det_ds

    def __len__(self):
        return self.samples_per_epoch
    
    def box_norm(self, x, ori_size):
        h, w = ori_size
        x = box_xyxy_to_cxcywh(x)
        x = x / torch.tensor([w, h, w, h], dtype=torch.float32)
        
        return x


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
        ds = random.randint(0, len(self.ground_det_ds_list) - 1)
        ds = self.ground_det_ds_list[ds]
        ground_det_ds = self.ground_det_data[ds]
        images = ground_det_ds["images"]
        annotations = ground_det_ds["ann"]
        
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        anns = annotations[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        if len(anns) > self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(anns))), size=self.num_classes_per_sample, replace=False)
        else:
            sampled_inds = list(range(len(anns)))
        sampled_anns = np.vectorize(anns.__getitem__)(sampled_inds).tolist()
        sampled_classes = []
        boxes = []
        questions = []
        answers = []
        if ds == 'genome':
            for cur_ann in sampled_anns:
                if len(cur_ann['bbox']) != 0:
                    sampled_classes.append(cur_ann['expression'])
                    boxes.append(self.box_norm(torch.tensor(cur_ann['bbox'], dtype=torch.float32), ori_size))

            for text in sampled_classes:
                text = text.strip()
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))
                answers.append(random.choice(self.answer_list))

        elif ds == 'flickr':
            sampled_true_classes = []
            sampled_box = []
            for cur_ann in sampled_anns:
                if len(cur_ann['cat']) != 0:
                    sampled_classes.append(cur_ann['sentence'])
                    sampled_true_classes.append(cur_ann['cat'])
                    sampled_box.append(cur_ann['boxes'])

            

            for kk in range(len(sampled_classes)):
                text = sampled_classes[kk].strip()
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text.lower()))
                this_sampled_true_classes = sampled_true_classes[kk]
                this_sampled_box = sampled_box[kk]
                this_answer = ''
                this_answer += random.choice(self.det_answer_list)
                for ii in range(len(this_sampled_true_classes)):
                    cur_box_list = this_sampled_box[ii]
                    ins_num = len(cur_box_list)
                    for jj in range(ins_num):
                        boxes.append(self.box_norm(torch.tensor(cur_box_list[jj], dtype=torch.float32), ori_size))
                    this_answer += this_sampled_true_classes[ii] + INS_TOKEN * ins_num + "\n"
                answers.append(this_answer)





        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]


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

        
        boxes = torch.stack(boxes, axis=0)

        
        masks = torch.rand(0, *ori_size)
        label = torch.ones(masks.shape[1], masks.shape[2]) * 255

        return (
            image_path,
            image,
            image_clip,
            conversations,
            boxes,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            ds,
        )

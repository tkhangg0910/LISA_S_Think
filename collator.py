from __future__ import annotations
import torch
from typing import Dict, List, Tuple
from model.MLLM.llava_med.constants import (
    IGNORE_INDEX, 
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN
)
import torch.nn.functional as F

CLASS2SEGTOKEN = {
    "GCA":    "[GCA_SEG]",
    "Koedam": "[KOEDAM_SEG]",
    "MTA":    "[MTA_SEG]",
}

ABLATION_MODES: Dict[str, Tuple[bool, bool, bool]] = {
    "full":       (True,  True,  True),
    "loc_only":   (True,  False, False),   # Localization Only
    "morph_only": (False, True,  False),   # Morphology Only
    "malig_only": (False, False, True),    # Malignancy Only
    "a1_a2":      (True,  True,  False),   # A1 + A2
    "a2_a3":      (False, True,  True),    # A2 + A3
    "a1_a3":      (True,  False, True),    # A1 + A3
}
 
 
class TracCoTCollatorAblation(TracCoTCollatorv2):

 
    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
        use_mm_start_end: bool = False,
        mode: Optional[str] = None,
        include_q1: Optional[bool] = None,
        include_q2: Optional[bool] = None,
        include_q3: Optional[bool] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            use_mm_start_end=use_mm_start_end,
        )
 
        if mode is not None:
            if mode not in ABLATION_MODES:
                raise ValueError(
                    f"Unknown mode='{mode}'. Available: {list(ABLATION_MODES)}"
                )
            self.include_q1, self.include_q2, self.include_q3 = ABLATION_MODES[mode]
        else:

            self.include_q1 = True if include_q1 is None else include_q1
            self.include_q2 = True if include_q2 is None else include_q2
            self.include_q3 = True if include_q3 is None else include_q3
 
        self.mode = mode or "custom"
 

    def _encode_q4_header_ids(self, q4_text, prepend_image):
        prefix = (
            f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
            if (self.use_mm_start_end and prepend_image)
            else (f"{DEFAULT_IMAGE_TOKEN}\n" if prepend_image else "")
        )
        q_text = f"{prefix}Q4: {q4_text}\n"
        return self.tokenizer(q_text, add_special_tokens=False)["input_ids"]
 

    def _encode_prefix(self, q4_text, q1_text, prepend_image):
        assert self.include_q1, "_encode_prefix chỉ được gọi khi include_q1=True"
        assert q1_text is not None, "Cần Q1 text để build bước localization"
 
        q_ids = self._encode_q4_header_ids(q4_text, prepend_image)
 
        a1_text = f"Q1: {q1_text}\nA1: "
        a1_ids = self.tokenizer(a1_text, add_special_tokens=False)["input_ids"]
 
        obj_text = "<obj>"
        obj_ids = self.tokenizer(obj_text, add_special_tokens=False)["input_ids"]
        assert len(obj_ids) == 1, "`<obj>` phải encode ra đúng 1 token"
 
        input_ids = q_ids + a1_ids + obj_ids
        labels = [IGNORE_INDEX] * len(q_ids) + a1_ids + obj_ids
        attn_mask = [1] * len(input_ids)
        return self._pad(input_ids, labels, attn_mask, self.max_length_prefix)
 
    def _encode_suffix_no_localization(
        self, q4_text, prepend_image, q2, a2, q3, a3, q4, a4, seg_str
    ):
        header_ids = self._encode_q4_header_ids(q4_text, prepend_image)
        input_ids = list(header_ids)
        labels    = [IGNORE_INDEX] * len(header_ids)
        attn_mask = [1] * len(header_ids)
 
        body = self._encode_suffix(q2, a2, q3, a3, q4, a4, seg_str)
        body_len = int(body["attention_mask"].sum().item())
        input_ids += body["input_ids"][:body_len].tolist()
        labels    += body["labels"][:body_len].tolist()
        attn_mask += [1] * body_len
 
        return self._pad(input_ids, labels, attn_mask, self.max_length_suffix)
 
    def __call__(self, batch: List[Dict]) -> Dict:
        images      = torch.stack([item["image"]      for item in batch])
        images_clip = torch.stack([item["image_clip"] for item in batch])
        images_dino = torch.stack([item["image_dino"] for item in batch])
        orig_sizes  = [item["orig_size"] for item in batch]
 
        max_n = max(item["bboxes"].shape[0] for item in batch)
        bboxes_list, bbox_labels_list, bbox_masks_list = [], [], []
        for item in batch:
            b   = item["bboxes"].clone().float()
            lbl = item["bbox_labels"].clone().long()
            n   = b.shape[0]
            if n < max_n:
                b   = torch.cat([b,   torch.zeros(max_n - n, 4, dtype=torch.float)], dim=0)
                lbl = torch.cat([lbl, torch.full((max_n - n,), -100, dtype=torch.long)],  dim=0)
            mask = torch.zeros(max_n, dtype=torch.bool)
            mask[:n] = True
            bboxes_list.append(b); bbox_labels_list.append(lbl); bbox_masks_list.append(mask)
 
        bboxes_tensor      = torch.stack(bboxes_list)
        bbox_labels_tensor = torch.stack(bbox_labels_list)
        bbox_masks_tensor  = torch.stack(bbox_masks_list)
 
        gt_masks = (torch.stack([item["gt_mask"] for item in batch])
                    if batch[0].get("gt_mask") is not None else None)
 
        prefix_list, suffix_list = [], []
        for sample in batch:
            conv          = sample["conversations"]
            prepend_image = conv[0].get("prepend_image", False)
            seg_classes   = sample.get("seg_classes", [])
            seg_str = "".join(self.class2segtoken[c] for c in seg_classes if c in self.class2segtoken)
 
            q2 = conv[1].get("Q2") if self.include_q2 else None
            a2 = conv[1].get("A2") if self.include_q2 else None
            q3 = conv[2].get("Q3") if self.include_q3 else None
            a3 = conv[2].get("A3") if self.include_q3 else None
            q4 = conv[3].get("Q4")
            a4 = conv[3].get("A4")
 
            if self.include_q1:
                q1 = conv[0].get("Q1")
                prefix_list.append(self._encode_prefix(q4, q1, prepend_image))
                suffix_list.append(self._encode_suffix(q2, a2, q3, a3, q4, a4, seg_str))
            else:
                suffix_list.append(
                    self._encode_suffix_no_localization(
                        q4, prepend_image, q2, a2, q3, a3, q4, a4, seg_str
                    )
                )
 
        def _stack(lst, key):
            return torch.stack([d[key] for d in lst])
 
        if self.include_q1:
            prefix_out = {
                "prefix_input_ids":       _stack(prefix_list, "input_ids"),
                "prefix_attention_masks": _stack(prefix_list, "attention_mask"),
                "prefix_labels":          _stack(prefix_list, "labels"),
            }
        else:
            # Không có bước localization -> không có prefix/<obj> gì cả.
            prefix_out = {
                "prefix_input_ids":       None,
                "prefix_attention_masks": None,
                "prefix_labels":          None,
            }
 
        return {
            "images":           images,
            "images_clip":      images_clip,
            "images_dino":      images_dino,
            "bboxes_gt":        bboxes_tensor,
            "bbox_labels_gt":   bbox_labels_tensor,
            "bbox_masks":       bbox_masks_tensor,
            "gt_masks":         gt_masks,
            **prefix_out,
            "suffix_input_ids":       _stack(suffix_list, "input_ids"),
            "suffix_attention_masks": _stack(suffix_list, "attention_mask"),
            "suffix_labels":          _stack(suffix_list, "labels"),
            "metas":             [s["meta"] for s in batch],
            "orig_sizes":        orig_sizes,
            "seg_classes_batch": [s["seg_classes"] for s in batch],
            "ablation_mode":     self.mode,
            "has_localization":  self.include_q1,
        }

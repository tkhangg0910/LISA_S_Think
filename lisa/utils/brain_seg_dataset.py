"""LISA datasets for tkhangg0910/brain_data_split_v1 (brain MRI atrophy segmentation).

Task, as specified: input is Q4, output is A4 plus three segmentations
(MTA, GCA, Koedam), with the reasoning chain distilled from Q1/A1..Q3/A3.

The parquet files are read in place -- nothing is derived to disk, and the
train/test split shipped in the repo is used exactly as-is.

WHY A FIXED THREE [SEG]
-----------------------
LISA asserts that the number of [SEG] tokens in an answer equals the number of
ground-truth masks for that image (model/LISA.py, `gt_mask.shape[0] ==
pred_mask.shape[0]`). Measured over the data:

    non-empty masks per row   train {0:244, 1:387, 2:368, 3:1}
                              test  {0:207, 1:202, 2:133, 3:0}

so a variable [SEG] scheme would emit 0-2 masks and leave the absent classes
with no prediction at all -- per-class IoU/DICE for MTA (empty in 93% of rows)
would then be undefined on most samples. Since the objective here is per-class
IoU/DICE for MTA, GCA and Koedam, this module always emits exactly three [SEG]
in the fixed order MTA -> GCA -> Koedam, and always supplies three ground-truth
masks (all-zero where the class is absent). Every class therefore has a
prediction on every sample and the metrics are well defined everywhere.
"""

import io
import random

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN

# fixed emission order; masks_list rows follow the same order
CLASSES = ["MTA", "GCA", "Koedam"]
CLASS_PHRASE = {
    "MTA": "medial temporal atrophy",
    "GCA": "global cortical atrophy",
    "Koedam": "posterior atrophy",
}

# columns actually needed -- the parquet has 85, most of them translations
COLUMNS = [
    "my_id", "Patient ID", "Status", "Q4", "A2", "A3", "A1",
    "image", "MTA_mask", "GCA_mask", "Koedam_mask",
]


def _decode(cell):
    """HF `image` cells are {'bytes':..., 'path':...}; return an RGB uint8 array."""
    if cell is None or not cell.get("bytes"):
        return None
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def make_square(image, mode="pad_square", bg=0):
    """Make `image` square so CLIPImageProcessor's center crop becomes a no-op.

    THE BUG THIS FIXES
    ------------------
    checkpoints/clip-vit-large-patch14/preprocessor_config.json sets
    do_center_crop=true, size=224. For a 496x248 slice that resizes the SHORT edge to
    224 and then crops 224x224 out of the middle -- the LLM branch sees only the middle
    50% of the long axis. Measured over the 1000 training rows, by lesion pixel:
        MTA     0.0% of lesion pixels lost,  0/69  rows lost entirely
        GCA     1.6%                         0/618
        Koedam 40.1%                        32/439   <-- the worst-scoring class
    SAM's branch is unaffected (ResizeLongestSide + bottom/right pad, no crop), which is
    why segmentation worked at all; the LLM simply never saw where posterior atrophy is.

    Feeding a square image makes resize-then-center-crop lossless. Aspect ratio is
    preserved rather than squashed: half the rows are 90-degree rotations of the other
    half, and an anisotropic squash would make those two orientations geometrically
    different instead of pure rotations.

    mode="pad_square"  letterbox the whole image into a square (default fix)
    mode="tight_pad"   crop to the brain's foreground bbox first, then letterbox --
                       spends more of the 224x224 on tissue (~58% vs 50% fill here)
    mode="center_crop" leave the image alone, i.e. reproduce the old buggy behaviour
    """
    if mode == "center_crop":
        return image
    img = image
    if mode == "tight_pad":
        grey = img[..., 0] if img.ndim == 3 else img
        ys, xs = np.where(grey > 10)
        if len(ys) and len(xs):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.full((side, side, img.shape[2]) if img.ndim == 3 else (side, side),
                  bg, dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def _decode_mask(cell):
    """Masks are soft-edged greyscale (values 0..199); binarise on >0."""
    if cell is None or not cell.get("bytes"):
        return None
    m = np.array(Image.open(io.BytesIO(cell["bytes"])).convert("L"))
    return (m > 0).astype(np.uint8)


NEUTRAL_ANSWER = (
    "Medial temporal atrophy: [SEG]. "
    "Global cortical atrophy: [SEG]. "
    "Posterior atrophy: [SEG]."
)


SEG_SENTENCE = (
    "Medial temporal atrophy: [SEG]. "
    "Global cortical atrophy: [SEG]. "
    "Posterior atrophy: [SEG]."
)


def render_a1(a1_raw, decimals=2):
    """A1 is a python-literal list of normalized [x1,y1,x2,y2] boxes with the class
    label stripped. Render it compactly -- full float repr costs ~250 extra tokens and
    blows collate_fn's 257-token training truncation, which would silently delete the
    trailing [SEG] and fire the assert at model/LISA.py:354."""
    import ast
    try:
        boxes = ast.literal_eval((a1_raw or "[]").strip())
    except Exception:
        return ""
    if not boxes:
        return "Regions of interest: none."
    f = lambda v: f"{round(float(v), decimals):g}"
    items = ", ".join("(" + ",".join(f(v) for v in b) + ")" for b in boxes)
    return f"Regions of interest: {items}."


REASONING_MODES = ("full", "localization_only", "morph_only", "malig_only")


def validate_reasoning_mode(mode, order, use_q1, neutral=False):
    if mode not in REASONING_MODES:
        raise ValueError(f"Unknown reasoning_mode: {mode}")
    if mode != "full" and (order != "reason_first" or use_q1 or neutral):
        raise ValueError("Ablation modes require reason_first, no use_q1 and no neutral_reasoning")


def build_answer(row, present, neutral=False, order="reason_first", use_q1=False,
                 reasoning_mode="full"):
    """Build the assistant turn. Always exactly three [SEG] in CLASSES order.

    `present` maps class -> bool, taken from the masks (not from A3): A3 and the
    mask annotations disagree on 40-75% of rows depending on the class, and the
    masks are the thing being scored, so they are the source of truth here.

    order="reason_first"  reasoning, then the three [SEG]   (the no-Q1 baseline)
    order="seg_first"     the three [SEG], then the reasoning as post-hoc explanation

    Why seg_first exists: attention is causal, so anything before a [SEG] is readable
    by the hidden state that produces its mask. Both the presence sentence AND A2 state
    which classes are present -- A2 does so in 287/300 sampled rows. With reason_first
    the model is handed the answer and never has to infer presence from the image, which
    is exactly why specificity measured 1.000 with the text and 0.002-0.058 without it.
    seg_first removes every textual cue from the mask path.
    """
    validate_reasoning_mode(reasoning_mode, order, use_q1, neutral)
    if reasoning_mode != "full":
        step = {"localization_only": "1", "morph_only": "2", "malig_only": "3"}[reasoning_mode]
        keys = (f"Q{step}", f"A{step}", "Q4", "A4")
        for key in keys:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row.get('my_id', '<unknown>')}: missing/empty {key} for {reasoning_mode}")
        return "\n".join(f"{key}: {row[key].strip()}" for key in keys) + "\n" + SEG_SENTENCE

    if order == "a4_seg":
        # Non-reasoning variant: input Q4, output A4 plus the three [SEG]. No A2
        # findings, no presence sentence, no A3 severity -- so nothing in the answer
        # states which markers are present and the masks must come from the image.
        # This is the like-for-like ablation of what the reasoning chain contributes.
        return f"Status ({(row['Status'] or '').strip()}). " + SEG_SENTENCE

    if neutral:
        # Strips every ground-truth statement, keeping only the three [SEG].
        # Used to measure how much of the score comes from the image rather than
        # from the reasoning text that teacher-forced evaluation hands the model.
        return NEUTRAL_ANSWER

    findings = (row["A2"] or "").strip()
    severity = (row["A3"] or "").strip().rstrip(".")
    status = (row["Status"] or "").strip()

    seen = ", ".join(CLASS_PHRASE[c] for c in CLASSES if present[c])
    absent = ", ".join(CLASS_PHRASE[c] for c in CLASSES if not present[c])
    obs = []
    if seen:
        obs.append(f"There is {seen}")
    if absent:
        obs.append(f"there is no {absent}")
    observation = "; ".join(obs) + "."

    parts = []
    if use_q1:
        loc = render_a1(row.get("A1"))
        if loc:
            parts.append(loc)
    if findings:
        parts.append(f"Findings: {findings}")
    parts.append(observation.capitalize() if not seen else observation)
    if severity:
        parts.append(f"Severity: {severity}.")
    parts.append(f"Therefore the pathological impact is Status ({status}).")

    if order == "seg_first":
        return " ".join([SEG_SENTENCE] + parts)
    return " ".join(parts + [SEG_SENTENCE])


class _BrainBase(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def _init_common(self, parquet_path, tokenizer, vision_tower, image_size):
        # keep the Arrow table rather than python objects: DataLoader workers fork,
        # and python refcounting would copy a few hundred MB per worker.
        columns = list(COLUMNS)
        if self.reasoning_mode != "full":
            step = {"localization_only": "1", "morph_only": "2", "malig_only": "3"}[self.reasoning_mode]
            columns += [f"Q{step}", "A4"]
        self.table = pq.read_table(parquet_path, columns=columns)
        self.n = self.table.num_rows
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        print(f"brain_seg: {self.n} rows from {parquet_path}")

    def _row(self, idx):
        return {k: v[0] for k, v in self.table.slice(idx, 1).to_pydict().items()}

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        return F.pad(x, (0, self.img_size - w, 0, self.img_size - h))

    neutral_reasoning = False
    answer_order = "reason_first"
    clip_view = "center_crop"
    use_q1 = False
    reasoning_mode = "full"

    def _build(self, idx):
        """Shared body: returns the 9-tuple LISA's collate_fn expects."""
        row = self._row(idx)
        image = _decode(row["image"])

        masks = []
        present = {}
        for c in CLASSES:
            m = _decode_mask(row[f"{c}_mask"])
            if m is None:
                m = np.zeros(image.shape[:2], dtype=np.uint8)
            if m.shape != image.shape[:2]:
                m = cv2.resize(m, (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            present[c] = bool(m.any())
            masks.append(m)

        # CLIP branch (224) and SAM branch (1024) see the same image
        image_clip = self.clip_image_processor.preprocess(
            make_square(image, self.clip_view), return_tensors="pt")["pixel_values"][0]
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
        )

        question = DEFAULT_IMAGE_TOKEN + "\n" + (row["Q4"] or "").strip()
        answer = build_answer(row, present, neutral=self.neutral_reasoning,
                              order=self.answer_order, use_q1=self.use_q1,
                              reasoning_mode=self.reasoning_mode)

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        masks = torch.from_numpy(np.stack(masks, axis=0))          # [3, H, W]
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            row["my_id"],
            image_sam,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            [question],
            list(CLASSES),
        )


class BrainSegDataset(_BrainBase):
    """Training split. Like the other LISA datasets, __len__ is a virtual epoch
    length and __getitem__ ignores idx (HybridDataset always asks for index 0)."""

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
        brain_seg_data="data/brain_data_split_v1/data/train-00000-of-00001.parquet",
        answer_order="reason_first",
        clip_view="center_crop",
        use_q1=False,
        reasoning_mode="full",
    ):
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.answer_order = answer_order
        self.clip_view = clip_view
        self.use_q1 = use_q1
        self.reasoning_mode = reasoning_mode
        validate_reasoning_mode(reasoning_mode, answer_order, use_q1, self.neutral_reasoning)
        self._init_common(brain_seg_data, tokenizer, vision_tower, image_size)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self._build(random.randint(0, self.n - 1))


class BrainSegValDataset(_BrainBase):
    """Held-out split, iterated in order. Returns inference=True so that
    train_ds.py's validate() takes the teacher-forced path."""

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        brain_seg_data="data/brain_data_split_v1/data/test-00000-of-00001.parquet",
        image_size=1024,
        neutral_reasoning=False,
        answer_order="reason_first",
        clip_view="center_crop",
        use_q1=False,
        reasoning_mode="full",
    ):
        self.neutral_reasoning = neutral_reasoning
        self.answer_order = answer_order
        self.clip_view = clip_view
        self.use_q1 = use_q1
        self.reasoning_mode = reasoning_mode
        validate_reasoning_mode(reasoning_mode, answer_order, use_q1, self.neutral_reasoning)
        self._init_common(brain_seg_data, tokenizer, vision_tower, image_size)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return *self._build(idx), True

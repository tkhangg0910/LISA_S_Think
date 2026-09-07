# LISA-7B trên HiPerGator — setup

Mọi thứ (conda env, cache pip/HF/torch/triton, weights, dataset, log, output)
đều nằm **trong** `/blue/rc-rse/vanminh.nguyen/NguyenDo/Experiment_ICLR_2027/LISA`.
Không có gì ghi ra `$HOME` ngoài dòng đăng ký env của conda trong
`~/.conda/environments.txt` (vài byte, conda luôn ghi, không tắt được).

---

## 1. Tại sao không dùng được `requirements.txt` gốc

Cluster này chỉ có 2 loại GPU:

| Partition | GPU | Compute cap. | VRAM | Dùng cho |
|---|---|---|---|---|
| `hpg-turin`, `hwgui` | L4 | sm_89 | 24 GB | inference |
| `hpg-b200` | B200 | sm_100 | 180 GB | training + inference |

`requirements.txt` của LISA pin `torch==1.13.1+cu117`. CUDA 11.7 **không có
kernel cho sm_89 lẫn sm_100** → không chạy được trên bất kỳ GPU nào ở đây.
Vì vậy có **hai env**, khác nhau đúng ở build CUDA của torch:

| Env | torch | CUDA | GPU |
|---|---|---|---|
| `env/` | 2.1.2 | cu121 | L4 (sm_89) |
| `env-b200/` | 2.7.1 | cu128 | B200 (sm_100) |

Cả hai đều giữ **`transformers==4.31.0` + `peft==0.4.0`**. Đây là ràng buộc cứng:
`model/llava/` và `model/LISA.py` viết theo API 4.31 (chữ ký `forward` tự chế của
`LlavaLlamaForCausalLM`, `prepare_inputs_for_generation`, truyền `images=` xuyên
qua `generate`). Nâng transformers = phải port lại code model, không phải bump version.

### Đã kiểm chứng trên GPU thật (`scripts/smoke_test.py`)

| | L4 | B200 |
|---|---|---|
| `torch.cuda.get_arch_list()` | có `sm_86` (chạy được sm_89) | `[sm_75, sm_80, sm_86, sm_90, **sm_100**, sm_120, compute_120]` |
| import transformers 4.31 / peft / deepspeed | ✅ | ✅ (torch 2.7.1 **không** dính lỗi pytree) |
| SAM ViT-H bf16 forward 1024×1024 | ✅ 2.82 GiB | ✅ 2.82 GiB |

Vì torch 2.7.1 chạy transformers 4.31 sạch, shim `setup/torch_compat_b200.py`
**không được cài**. Nó nằm đó phòng khi sau này nâng torch lên ≥2.8 và vỡ ở
`torch.load(weights_only=True)` hoặc `_pytree._register_pytree_node` — lúc đó
copy nó thành `env-b200/lib/python3.10/site-packages/sitecustomize.py`.

### Hai pin không hiển nhiên

- **`setuptools<81` ở env L4.** torch 2.1.2 `utils/cpp_extension.py` còn
  `from pkg_resources import packaging`, mà setuptools 81 đã bỏ `pkg_resources`
  → `import transformers` chết ngay. env B200 (torch 2.7.1) không dính.
- **`scikit-image`.** `utils/refer.py` import `skimage.io`, mà `utils/dataset.py`
  import `refer` → thiếu nó là train/eval chết. Không có trong `requirements.txt` gốc.

---

## 2. Dùng hàng ngày

```bash
cd /blue/rc-rse/vanminh.nguyen/NguyenDo/Experiment_ICLR_2027/LISA
source env.sh                        # tự chọn env theo GPU đang thấy
LISA_ENV_NAME=env-b200 source env.sh # ép chọn env B200
```

`env.sh` set toàn bộ `HF_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME`, `TRITON_HOME`,
`TMPDIR`, `CONDA_PKGS_DIRS`… về trong project rồi `conda activate`.

---

## 3. Cây thư mục

```
LISA/
├── env/                    conda env cho L4        (torch 2.1.2+cu121)
├── env-b200/               conda env cho B200      (torch 2.7.1+cu128)
├── .cache/                 pip / huggingface / torch / triton / tmp
├── checkpoints/            LISA-7B-v1, LISA-7B-v1-explanatory, sam_vit_h
├── dataset/reason_seg/ReasonSeg/{train,val,test,explanatory}
│                           ReasonSeg: 239 / 200 / 779 ảnh
│                           (bản tải về thiếu cấp `ReasonSeg/`, script tự sửa)
├── runs/                   checkpoint + tensorboard khi train
├── logs/                   log SLURM
├── vis_output/             mask + overlay
├── env.sh                  <-- source cái này
├── setup/                  script cài đặt
├── slurm/                  job script
└── scripts/                infer.py, smoke_test.py
```

---

## 4. Cài lại từ đầu

```bash
sbatch slurm/setup_env.sbatch        # env L4
sbatch slurm/setup_env_b200.sbatch   # env B200
sbatch slurm/setup_weights.sbatch    # LISA-7B x2 + CLIP + SAM ViT-H  (~35 GB)
sbatch slurm/setup_reasonseg.sbatch   # ReasonSeg  (~2 GB)
```

Cài đặt chạy trong SLURM chứ không chạy trên login node: login node giết tiến
trình tải torch 2.2 GB giữa chừng.

**Lưu ý mạng — đọc trước khi tải thêm model nào:** trên compute node,
`curl` tới huggingface.co mất ~0.3 s, nhưng HTTP stack của `huggingface_hub`
mất ~80 s mỗi kết nối và `snapshot_download()` thì **treo hẳn** (0 byte sau 4 phút,
thử lại 2 lần đều thế). Vì vậy `setup/02_download_weights.sh` tải bằng `curl`
thẳng vào URL `/resolve/main/` — nhanh, resume được, và nhìn thấy tiến độ.
Đừng thay bằng `snapshot_download` / `hf_transfer`.

CLIP vision tower được để thành thư mục thật `checkpoints/clip-vit-large-patch14`,
và script sửa `config.json` của 2 checkpoint LISA để trỏ `vision_tower` /
`mm_vision_tower` vào đó — nhờ vậy lúc chạy không cần resolve
`openai/clip-vit-large-patch14` qua mạng.

---

## 5. Kiểm tra env

```bash
sbatch slurm/smoke_l4.sbatch
sbatch slurm/smoke_b200.sbatch
# hoặc kèm checkpoint để chạy 1 query thật:
sbatch slurm/smoke_b200.sbatch --version checkpoints/LISA-7B-v1-explanatory
```

`scripts/smoke_test.py` kiểm 5 tầng riêng biệt (torch/GPU arch → import stack →
import module LISA → matmul + SAM ViT-H forward → 1 query end-to-end), báo lỗi ở
đúng tầng nào hỏng.

---

## 6. Inference (reasoning segmentation)

`chat.py` gốc là REPL tương tác nên vô dụng trong batch job. Dùng `scripts/infer.py`:

```bash
# 1 câu hỏi
sbatch slurm/infer_l4.sbatch \
    --image imgs/example1.jpg \
    --prompt "Where can the driver see the car speed in this image? Please output segmentation mask."

# nhiều câu, mỗi dòng 1 JSON {"image": ..., "prompt": ..., "id": ...}
sbatch slurm/infer_b200.sbatch --jsonl queries.jsonl --out_dir vis_output/run1
```

Kết quả: `<out_dir>/<id>_mask_0.png`, `<id>_overlay_0.jpg`, và `results.jsonl`.
Một câu trả lời có thể chứa **nhiều `[SEG]`** (vd "Sure, [SEG]. In the image, the
sitting presidents are [SEG]."); `infer.py` lưu hết thành `_mask_0_0`, `_mask_0_1`…
(`chat.py` gốc chỉ lưu cái đầu tiên và bỏ im lặng phần còn lại).

Chọn checkpoint bằng biến `LISA_CKPT`, mặc định là `LISA-7B-v1-explanatory`:

- `LISA-7B-v1-explanatory` — trả lời có giải thích (đúng nghĩa *reasoning*)
- `LISA-7B-v1` — chỉ ra mask

**Kỳ vọng chất lượng:** ảnh mẫu trong `vis_output/*.jpg` của repo là output của
**LISA-13B**, không phải 7B. Khi chạy 7B, câu referring dễ ("segment the steering
wheel") ra mask rất chuẩn, câu reasoning khó thì hên xui — vd
"Where can the driver see the car speed" trên `imgs/example1.jpg` thì 7B segment
nhầm sang kính chắn gió, trong khi 13B ra đúng cụm đồng hồ. Đừng nhầm cái đó với
lỗi cài đặt; muốn đối chứng nhanh thì chạy `queries_debug.jsonl`.

Trên L4 (24 GB), 7B bf16 tốn ~17 GB weight + SAM encoder ở 1024×1024. Nếu OOM,
thêm `--load_in_8bit` (chỉ có ở env L4, env B200 không cài bitsandbytes vì không cần).

---

## 7. Eval trên ReasonSeg val

```bash
sbatch slurm/eval_reasonseg_test_b200.sbatch   # test split (779 ảnh) <-- dùng cái này
sbatch slurm/eval_reasonseg_b200.sbatch        # val split (200 ảnh)  <-- xem cảnh báo dưới
```

Chạy `train_ds.py --eval_only`, in ra **gIoU / cIoU**.

🚫 **Đừng benchmark `LISA-7B-v1` trên split `val`.** `v1` được train trên cả
`train+val` ([README.md](README.md) mục Validation), nên đo trên `val` là đo lại
dữ liệu huấn luyện. Dùng split `test` (779 ảnh, model chưa thấy), hoặc model `v0`
với code legacy (`git checkout 0e26916`).

Số đo thực tế của `LISA-7B-v1-explanatory` trên máy này:

| split | gIoU | cIoU | |
|---|---|---|---|
| `val` (200 ảnh) | 0.9463 | 0.9712 | ❌ rò rỉ dữ liệu train — bỏ qua |
| `test` (779 ảnh) | **0.4630** | **0.4888** | ✅ khớp khoảng paper |

Dùng bảng này làm mốc: nếu sau khi anh sửa gì đó mà `test` lệch nhiều so với
0.46/0.49 thì là do thay đổi của anh, không phải do môi trường.

⚠️ `train_ds.py` dựng `HybridDataset` (tập **train**) **trước** khi kiểm `--eval_only`,
nên dù chỉ eval nó vẫn cố khởi tạo ADE20K / COCO-Stuff / refCOCO và chết vì thiếu thư mục.
Vì thế job eval truyền `--dataset="reason_seg" --sample_rates="1"` để giới hạn
vào đúng dữ liệu có trên đĩa.

⚠️ Đường này là **teacher-forced**: `"[SEG]."` đã nằm sẵn trong prompt
(xem `ValDataset.__getitem__`), nên nó đo chất lượng mask chứ **không** đo việc
LLM có tự quyết định sinh `[SEG]` hay không. Muốn đo end-to-end thật thì phải
đi qua `model.evaluate()` như `scripts/infer.py`.

---

## 8. Training / fine-tune (chỉ B200)

```bash
sbatch slurm/train_b200.sbatch                       # 2×B200, ReasonSeg only
sbatch slurm/train_b200.sbatch --epochs 20 --lr 1e-4
```

🚨 **Bẫy lớn nhất của repo này.** `train_ds.py` gọi `initialize_lisa_modules()`
**sau** `from_pretrained()`, và hàm đó không *load* gì cả — nó **dựng mới**:

```python
self.visual_model    = build_sam_vit_h(self.vision_pretrained)     # SAM gốc
self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])    # random init
```

Đúng khi bắt đầu từ LLaVA (checkpoint không có SAM/projection), nhưng khi fine-tune
**từ checkpoint LISA** thì nó **xoá sạch mask decoder và lớp chiếu `[SEG]` đã train**
— không có cảnh báo nào, chỉ thấy loss bắt đầu cao. Tôi đã thêm cờ opt-in
`--restore_lisa_modules` (mặc định TẮT, giữ nguyên hành vi gốc) để nạp lại
`visual_model.*` + `text_hidden_fcs.*` từ checkpoint; `slurm/train_b200.sbatch`
bật sẵn cờ này. Bỏ cờ đi nếu bắt đầu từ LLaVA. Xem
[scripts/restore_lisa_modules.py](scripts/restore_lisa_modules.py).

Mặc định fine-tune từ `checkpoints/LISA-7B-v1` trên `reason_seg` — vì đó là dataset
duy nhất có sẵn ở local. Muốn đúng công thức paper thì cần thêm ADE20K / COCO-Stuff /
Mapillary / PACO-LVIS / PASCAL-Part / refCOCO* / LLaVA-150K (xem `README.md` gốc,
mục *Training Data Preparation*), rồi đổi:

```
--dataset="sem_seg||refer_seg||vqa||reason_seg" --sample_rates="9,3,3,1"
```

Train từ LLaVA gốc (thay vì từ LISA-7B) cần merge delta weight
`liuhaotian/LLaVA-Lightning-7B-delta-v1-1` với LLaMA-7B base — LLaMA-7B không có
trên HF công khai nên đường này phải tự lo weights.

Sau khi train xong:

```bash
cd runs/<exp_name>/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
cd $LISA_ROOT && python merge_lora_weights_and_save_hf_model.py \
    --version=checkpoints/LISA-7B-v1 \
    --weight=runs/<exp_name>/pytorch_model.bin \
    --save_path=checkpoints/<ten-model-moi>
```

---

## 9. Những chỗ dễ vỡ

0. **`from_pretrained(..., vision_tower=...)` là config override, không phải kwarg thường.**
   `transformers` lấy mọi kwarg lạ trùng tên field trong config để **ghi đè config**.
   Nên `--vision-tower` (mặc định `openai/clip-vit-large-patch14` trong `chat.py`,
   `app.py`, `train_ds.py`) sẽ **đè mất** đường dẫn CLIP local đã vá trong
   `config.json`, và model đi gọi Hub → chết dưới `HF_HUB_OFFLINE=1`.
   `scripts/infer.py` đã set mặc định thành `checkpoints/clip-vit-large-patch14`.
   Nếu chạy `train_ds.py` / `chat.py` trực tiếp thì **phải** truyền tay:
   `--vision-tower=$LISA_ROOT/checkpoints/clip-vit-large-patch14`.
1. **Con số 255 hard-code** trong `model/LISA.py` (bù cho 1 token `<image>` nở
   thành 256 patch). Đổi CLIP encoder / độ phân giải / dùng >1 ảnh là gãy ngay.
2. **`config.use_cache = False`** bị ép cứng trong `LisaModel.__init__`. Đây là
   lý do `evaluate()` lấy được hidden state của cả chuỗi (mỗi bước generate tính
   lại từ đầu) — và cũng là lý do inference chậm. Bật lại KV-cache là hỏng logic
   trích `[SEG]`.
3. **`gradio` không được cài.** `app.py` cần gradio 3.39 (pydantic v1) còn
   deepspeed cần pydantic v2 → xung đột. Cần demo web thì tạo env riêng.
4. **`flash-attn` không cài** — `train_ds.py` không import nó, chỉ
   `model/llava/train/train_mem.py` dùng, mà file đó không nằm trong luồng LISA.
5. **`HF_HUB_OFFLINE=1`** được bật trong mọi job GPU. Nếu thêm model mới, nhớ tải
   trước trên `hpg-default` rồi mới chạy job GPU.

---

## 10. Brain MRI (brain_data_split_v1) — segmentation fine-tune

Task: input `Q4`, output `A4` + 3 mask (MTA, GCA, Koedam), reasoning chưng cất từ
`A2`/`A3`. Dùng **đúng split có sẵn** trong repo HF, không chia lại.

```bash
sbatch slurm/train_brain_b200.sbatch --epochs 40 --exp_name my-run
bash   scripts/export_trained_model.sh my-run          # ZeRO -> fp32 -> merge LoRA
python scripts/eval_brain_seg.py --version runs/my-run/hf_model
```

### Ba quyết định thiết kế, và số liệu đứng sau

1. **Luôn sinh đúng 3 `[SEG]`** (thứ tự cố định MTA → GCA → Koedam), GT luôn 3 mask,
   rỗng nếu lớp vắng. Số mask khác rỗng mỗi ảnh đo được:
   train `{0:244, 1:387, 2:368, 3:1}`, test `{0:207, 1:202, 2:133, 3:0}`.
   Nếu để `[SEG]` biến thiên, MTA (rỗng 93% số dòng) gần như không có prediction nào
   và IoU/DICE của nó vô định trên phần lớn mẫu.
2. **`dice_loss_weight` 0.5 → 2.0.** Tổn thương chiếm 0.5–5.6% diện tích ảnh
   (MTA median 0.53%, Koedam 1.27%, GCA 5.63%); ở mức đó BCE đơn thuần kéo về "đoán rỗng".
3. **Presence trong reasoning lấy từ mask, không từ `A3`.** `A3` lệch mask 40–75%
   tuỳ lớp (vd GCA `=0` mà vẫn có mask: 142 dòng). Mask là thứ bị chấm điểm.

### Hai điều làm con số lạc quan hơn thực tế

- **Split rò rỉ.** Cả 3 bệnh nhân test đều có trong train (9 bệnh nhân tổng cộng), và
  58/174 lát cắt test xuất hiện trong train chỉ khác góc xoay `mpr-N`. Đây là con số
  so sánh nội bộ giữa các cấu hình, **không phải** generalization.
- **Chọn checkpoint trên chính tập test.** `train_ds.py` lưu checkpoint khi gIoU cải
  thiện, mà val ở đây chính là test — không có tập nào khác nếu không chia lại.

### Bug đã phải vá để đường này chạy được

`model/LISA.py` chỉ gán `ce/dice/bce_loss_weight` trong nhánh
`if not hasattr(config, "train_mask_decoder")`, tức **chỉ khi khởi từ LLaVA**.
Fine-tune từ checkpoint LISA rơi vào `else` và crash ở backward đầu tiên với
`AttributeError: 'LISAForCausalLM' object has no attribute 'ce_loss_weight'`.
Đã chuyển 3 dòng gán ra ngoài if/else. Cùng với `--restore_lisa_modules` (mục 8),
đây là hai lỗi cùng chặn đúng một use case: fine-tune tiếp từ checkpoint LISA.

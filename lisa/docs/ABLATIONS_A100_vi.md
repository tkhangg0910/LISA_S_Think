# Một epoch LISA 13B trên RunPod, 1 × A100 80 GB

Chạy trong Python environment đã cài dependencies tương thích của LISA (xem
`setup/requirements-hpg.txt`). Script không gọi module/conda của HiPerGator.
Không chạy setup B200 chỉ vì script cũ dùng B200.

Từ `/workspace/LISA_S_Think/lisa`:

```bash
bash scripts/train_ablation_1epoch.sh morph_only
# Sau khi run trên xong, nếu muốn đo mode còn lại:
bash scripts/train_ablation_1epoch.sh malig_only
```

Mặc định weights nằm trong `lisa/checkpoints/`, parquet trong
`lisa/data/brain_data_split_v1/data/`. Có thể đặt biến `LISA_BASE`, `CLIP_PATH`,
`SAM_PATH`, `BRAIN_DATA_DIR` để dùng đường dẫn sẵn có. Script kiểm tra đường dẫn,
sau đó kiểm tra target và collator trên toàn bộ train/test bằng tokenizer thật
trước khi load model. Annotation thiếu/rỗng hoặc truncation mất SEG sẽ báo lỗi.

Mỗi run dùng GPU 0, batch 1 × accumulation 20 × 50 optimizer steps = 1.000 lượt
lấy mẫu (sampling có hoàn lại, không đảm bảo mỗi ảnh được lấy đúng một lần).
Train 1 epoch rồi validation toàn bộ split test theo teacher forcing hiện có.
Không thay đổi kiến trúc hay cách tính validation gIoU/cIoU.

Giữ recipe LoRA rank 8, bf16, 8 blocks SAM cuối, max_length 1024, CE/BCE/Dice
weights 1/2/2. Đây là cấu hình cần đo trên A100; chưa đảm bảo vừa VRAM bằng một
GPU run thực tế. Script một epoch là benchmark; scheduler cũng có horizon một
epoch, vì vậy không dùng checkpoint này làm khởi đầu chính thức cho run 40 epochs.

Hai flow target: `Q2 A2 Q4 A4 SEG` hoặc `Q3 A3 Q4 A4 SEG` trong một assistant turn;
user chỉ thấy ảnh + Q4. Dùng A4 nguyên văn. Không A1, không câu presence từ masks.
SEG luôn gồm ba token, thứ tự MTA, GCA, Koedam.

Kết quả: `logs/<exp>.log`, `runs/<exp>/run_config.json`,
`runs/<exp>/epoch_timing.jsonl`; checkpoint theo quy tắc best validation hiện tại.
Dòng `EPOCH_TIMING` có giây train, validation, checkpoint và peak VRAM allocated/
reserved của PyTorch (không phải toàn bộ VRAM nvidia-smi). Timing không bao gồm
preflight/load model/export. Khi đo run dài, epoch đầu có thể chậm do warmup.

Mỗi lần chạy tạo tên mới để tránh auto-resume. `EXP_NAME` tùy chỉnh phải chưa tồn
 tại. Khi có hai GPU, đặt `GPU_ID=1 MASTER_PORT=29502` cho run thứ hai, tên run riêng.

Eval checkpoint HF đã export bằng `scripts/eval_brain_seg.py` phải truyền
`--reasoning_mode morph_only` hoặc `--reasoning_mode malig_only` đúng với lúc train.

## Localization only

Mode `localization_only`: user Image + Q4; assistant Q1 A1 Q4 A4, rồi ba SEG theo thứ tự MTA/GCA/Koedam. Dùng A1 nguyên văn, không render lại bbox. Không truyền `--use_q1`.

`bash scripts/train_ablation_40epochs.sh localization_only` chạy mới 40 epochs, giữ recipe ablation nhưng max_length 2048 do A1 dài (preflight: train 1180, test 1160 text tokens; thêm 255 token ảnh). Validation sau mỗi epoch.

`bash scripts/eval_localization_a100.sh <exp_name>` export best checkpoint rồi eval teacher forcing trên test, max_length 2048. Báo `OVERALL.present.DICE` và `OVERALL.present.gIoU`.

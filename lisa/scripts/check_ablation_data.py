"""Validate every target and collator label on CPU before loading LISA 13B."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from model.llava import conversation as conversation_lib
from utils.brain_seg_dataset import build_answer
from utils.dataset import collate_fn

p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
p.add_argument('--data', required=True)
p.add_argument('--mode', required=True, choices=['localization_only', 'morph_only', 'malig_only'])
p.add_argument('--model_max_length', type=int, default=1024)
a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.version, use_fast=False, model_max_length=a.model_max_length, padding_side='right')
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_tokens('[SEG]')
tokenizer.add_tokens(['<im_start>', '<im_end>'], special_tokens=True)
seg_id = tokenizer('[SEG]', add_special_tokens=False).input_ids
assert len(seg_id) == 1
conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v1']
step = {'localization_only': '1', 'morph_only': '2', 'malig_only': '3'}[a.mode]
for split in ['train', 'test']:
    rows = pq.read_table(Path(a.data) / f'{split}-00000-of-00001.parquet',
                         columns=['my_id', f'Q{step}', f'A{step}', 'Q4', 'A4']).to_pylist()
    longest = 0
    for row in rows:
        answer = build_answer(row, {}, reasoning_mode=a.mode)
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        question = '<image>\n' + row['Q4'].strip()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        # Only collator text processing is tested; dummy image/mask tensors stay on CPU.
        sample = (row['my_id'], torch.zeros(3, 1, 1), torch.zeros(3, 1, 1),
                  [conv.get_prompt()], torch.zeros(3, 1, 1), torch.zeros(1, 1),
                  (1, 1), [question], ['MTA', 'GCA', 'Koedam'], False)
        batch = collate_fn([sample], tokenizer=tokenizer)
        ids, labels = batch['input_ids'][0], batch['labels'][0]
        assert (ids == seg_id[0]).sum().item() == 3, row['my_id']
        assert (labels == seg_id[0]).sum().item() == 3, row['my_id']
        assert (labels == -100).any(), row['my_id']
        supervised = tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
        for key in [f'Q{step}', f'A{step}', 'Q4', 'A4']:
            assert f'{key}:' in supervised, (row['my_id'], key)
        longest = max(longest, ids.numel())
    print(f'PREFLIGHT {split}: {len(rows)} rows OK, max text tokens={longest}, budget={a.model_max_length - 255}', flush=True)
print('Example assistant target:\n' + answer)

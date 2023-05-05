# for debug
#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 25009 main.py  --debug  --batch_size 32

python -m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 29108 \
main.py \
--epochs 400 \
--work_dir 'ckpts/3layer_emb480_full' \
--lr 2e-4 \
--nlayers 3 \
--warmup_ratio 0.0 \
--batch_size 32 \
--seed 22222222

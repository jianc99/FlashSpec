export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=48
# export ENABLE_INTRA_NODE_COMM=1
# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/longspec_benchmark.py --B 2 --target ../FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model ../FastHesse/checkpoints/JackFram/llama-68m/model.pth --rank_group 0 --draft_ranks 0 --gamma 1 --prefix_len 4000 --gen_len 90 --benchmark --printoutput --streamingllm_budget 256 --compile
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 tests/longspec_benchmark.py --B 128 --target checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model checkpoints/TinyLlama/TinyLlama_v1.1/model.pth --rank_group 0 1 2 3 --draft_ranks 0 1 2 3 --gamma 1 --prefix_len 4000 --gen_len 90 --printoutput --streamingllm_budget 256 --benchmark --compile
# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/selfspec_benchmark.py --B 1 --model ../FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --rank_group 0 --gamma 1 --prefix_len 4000 --gen_len 90 --printoutput --streamingllm_budget 512 --benchmark

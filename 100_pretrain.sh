export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export DEEPSPEED_TIMEOUT=7200
export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

deepspeed --no_local_rank src/run_clm.py hf_config.json --deepspeed --deepspeed_config ds_config_zero.json

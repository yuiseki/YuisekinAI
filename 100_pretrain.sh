export TORCH_DISTRIBUTED_TIMEOUT=7200000 # 2 時間
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export DEEPSPEED_TIMEOUT=7200000 # 2 時間
export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_IFNAME=lo

deepspeed --no_local_rank src/run_clm.py hf_config.json --deepspeed --deepspeed_config ds_config_zero.json

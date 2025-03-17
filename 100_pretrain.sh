export TORCH_DISTRIBUTED_TIMEOUT=3600000
export NCCL_DEBUG=INFO

deepspeed --no_local_rank src/run_clm.py hf_config.json --deepspeed --deepspeed_config ds_config_zero.json

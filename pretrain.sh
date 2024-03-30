deepspeed --no_local_rank src/run_clm.py --output_dir checkpoints/YuisekinAI-mistral-300M-FA2 hf_config.json --deepspeed --deepspeed_config ds_config_zero.json

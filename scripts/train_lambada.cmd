@echo off

python scripts/lambada/train_cls.py --train_data_path %1 --val_data_path %1 --output_dir ./model/lambada/cls --device cpu --num_epoch 2 & \
python scripts/lambada/data_processing.py --data_path %1 --output_dir ./test/res/text & \
python scripts/lambada/run_clm.py --tokenizer_name ./model/lambada/cls --model_name_or_path gpt2 --model_type gpt2 --train_file ./test/res/text/mlm_data.txt --output_dir ./model/lambada/gen --do_train --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --save_steps=10000 --num_train_epochs 2
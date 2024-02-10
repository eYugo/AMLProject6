$task_name=$Args[0]

py main.py --experiment=$task_name --experiment_name=$task_name/cartoon/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'cartoon'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1

py main.py --experiment=$task_name --experiment_name=$task_name/sketch/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'sketch'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1

py main.py --experiment=$task_name --experiment_name=$task_name/photo/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'photo'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1
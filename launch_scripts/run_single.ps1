$task_name=$Args[0]
$target_domain=$Args[1]


py main.py --experiment=$task_name --experiment_name=$task_name/$target_domain/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '$target_domain'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 > output.txt
$target_domain=$Args[0]

py main.py --experiment=domain_adaptation --experiment_name=domain_adaptation_v2/$target_domain/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '$target_domain'}" --batch_size=64 --num_workers=5 --grad_accum_steps=2

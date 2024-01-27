$target_domain=$Args[0]
$experiment=$Args[1]

Write-Host $target_domain
Write-Host $experiment

if (-Not $experiment){
    $experiment = 'baseline'
}

Write-Host $target_domain
Write-Host $experiment

py main.py --experiment=$experiment --experiment_name=$experiment/$target_domain/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '$target_domain'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1
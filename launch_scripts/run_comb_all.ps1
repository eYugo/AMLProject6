$task_name=$Args[0]

$layers_combinations = @(
	"layer1.0.conv1",
	"layer1.0.conv2, layer2.0.conv2",
	"layer2.1.conv2, layer3.0.conv2",
	"layer3.0.downsample.1, layer4.0.conv2",
	"layer1.1.conv2, layer2.0.downsample.1, layer3.1.conv2, layer4.0.conv1"
)

foreach ($layer_comb in $layers_combinations) {

    $folder_name = $layer_comb.Replace(".", "_")

	$experiment_folder = $task_name + "_" + $folder_name

    py main.py --experiment=$task_name --experiment_name=$experiment_folder/"cartoon"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'cartoon'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb > output.txt

    py main.py --experiment=$task_name --experiment_name=$experiment_folder/"sketch"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'sketch'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb > output.txt

    py main.py --experiment=$task_name --experiment_name=$experiment_folder/"photo"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'photo'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb > output.txt
}
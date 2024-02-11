$task_name=$Args[0]

$layers_combinations = @(
	"layer1.0.conv1",
	# "layer1.0.bn1",
	# "layer1.0.relu",
	"layer1.0.conv2",
	# "layer1.0.bn2",
	"layer1.1.conv1",
	# "layer1.1.bn1",
	# "layer1.1.relu",
	"layer1.1.conv2",
	# "layer1.1.bn2"
	"layer2.0.conv1",
	# "layer2.0.bn1",
	# "layer2.0.relu",
	"layer2.0.conv2",
	# "layer2.0.bn2",
	"layer2.1.conv1",
	# "layer2.1.bn1",
	# "layer2.1.relu",
	"layer2.1.conv2",
	# "layer2.1.bn2",
	"layer3.0.conv1",
	# "layer3.0.bn1",
	# "layer3.0.relu",
	"layer3.0.conv2",
	# "layer3.0.bn2",
	"layer3.1.conv1",
	# "layer3.1.bn1",
	# "layer3.1.relu",
	"layer3.1.conv2",
	# "layer3.1.bn2",
	"layer4.0.conv1",
	# "layer4.0.bn1",
	# "layer4.0.relu",
	"layer4.0.conv2",
	# "layer4.0.bn2",
    "layer4.1.conv1",
	# "layer4.1.bn1",
	# "layer4.1.relu",
	"layer4.1.conv2"
	# "layer4.1.bn2"
)

foreach ($layer_comb in $layers_combinations) {

    $folder_name = $layer_comb.Replace(".", "_")

	$experiment_folder = $task_name + "_" + $folder_name

    py main.py --experiment=$task_name --experiment_name=$experiment_folder/"cartoon"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'cartoon'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb

    py main.py --experiment=$task_name --experiment_name=$experiment_folder/"sketch"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'sketch'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb

    py main.py --experiment=$task_name --experiment_name=$experiment_folder/"photo"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'photo'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb
}
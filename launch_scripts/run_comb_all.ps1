$task_name=$Args[0]

$layers_combinations = @(
	# "layer1",
	# "layer1.0",
	# "layer1.0.conv1",
	# "layer1.0.bn1",
	# "layer1.0.relu",
	# "layer1.0.conv2",
	# "layer1.0.bn2",
	# "layer1.1",
	# "layer1.1.conv1",
	# "layer1.1.bn1",
	# "layer1.1.relu",
	# "layer1.1.conv2",
	"layer1.1.bn2",
	# "layer2.0.bn1",
	# "layer3.0.bn1",
	# "layer4.0.bn1",
	# "layer2.0.bn2",
	# "layer3.0.bn2",
	# "layer4.0.bn2",
	# "layer2.0.conv2",
	# "layer3.0.conv2",
	# "layer4.0.conv2",
	# "layer2.1.bn1",
	# "layer3.1.bn1",
	# "layer4.1.bn1",
	# "layer2.1.bn2",
	# "layer3.1.bn2",
	# "layer4.1.bn2",
	# "layer2.1.conv2",
	# "layer3.1.conv2",
	# "layer4.1.conv2"
	# "layer2.1.bn2",
	# "layer1.1.bn2,layer3.0.bn1,layer2.1.conv2",
	# "layer1.1.bn2,layer3.0.bn1",
	# "layer2",
	# "layer2.0",
	# "layer2.0.conv1",
	# "layer2.0.bn1",
	# "layer2.0.conv2",
	# "layer2.0.bn2",
	# "layer2.0.downsample.0",
	# "layer2.0.downsample.1",
	# "layer2.1",
	# "layer2.1.conv1",
	# "layer2.1.bn1",
	"layer2.1.conv2",
	"layer2.1.bn2",
	# "layer3",
	# "layer3.0",
	# "layer3.0.conv1",
	"layer3.0.bn1",
	# "layer3.0.conv2",
	# "layer3.0.bn2",
	"layer3.0.downsample.0",
	# "layer3.0.downsample.1",
	# "layer3.1",
	# "layer3.1.conv1",
	# "layer3.1.bn1",
	# "layer3.1.conv2",
	# "layer3.1.bn2",
	# "layer4.0.downsample.0",
	"layer4.0.downsample.1",
	# "avgpool",
	# "layer3.0.bn1,layer4.0.bn2",
	# "layer3.0.bn1,layer2.1.conv2",
	# "layer3.0.bn1,layer1.1.bn2",
	# "layer3.0.bn1,layer4.0.downsample.0",
	# "layer3.0.bn1,layer3.0.downsample.0",
	# "layer2.1.conv2,layer4.0.bn2",
	# "layer2.1.conv2,layer1.1.bn2",
	# "layer2.1.conv2,layer4.0.downsample.0",
	# "layer2.1.conv2,layer3.0.downsample.0",
	# "layer4.0.downsample.1,layer1.1.bn2,layer3.0.bn1",
	# "layer4",
	# "layer3.0.bn1,layer4.0.downsample.1",
	# "layer2.1.conv2,layer4.0.downsample.1"
	"layer4.0.conv1",
	"layer4.1.conv1"
)

$topKvalue = @(
	0.001,
	0.01,
	0.7,
	1
)

foreach ($layer_comb in $layers_combinations) {
	foreach ($topK in $topKvalue) {

		$folder_name = $layer_comb.Replace(".", "_")
		$folder_name = $folder_name.Replace(",", "_")
		$top_k_folder=$topK -replace '\.', '_'
		$experiment_folder = $task_name + "_" + $folder_name+"_"+ $top_k_folder+"_"+"variation2_0_1"

    	py main.py --experiment=$task_name --experiment_name=$experiment_folder/"cartoon"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'cartoon'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb --topK=$topK

    	py main.py --experiment=$task_name --experiment_name=$experiment_folder/"sketch"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'sketch'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb --topK=$topK

    	py main.py --experiment=$task_name --experiment_name=$experiment_folder/"photo"/ --dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': 'photo'}" --batch_size=128 --num_workers=5 --grad_accum_steps=1 --layer_list=$layer_comb --topK=$topK
	}

}
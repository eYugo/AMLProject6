# AML 2023/2024 Project - Activation Shaping for Domain Adaptation
Official repository for the "Activation Shaping for Domain Adaptation" project - Advanced Machine Learning Course 2023/2024 @ PoliTo

## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.4, installed and correctly configured on your system. 

(The base code of this project has been produced using CUDA 11.4 and Python 3.10.9)

Once you have properly setup everything, make sure you are in the correct directory and run from the command line:
```bash
pip install -r requirements.txt
```

### Dataset
1. Download the PACS dataset from the portal of the course in the "project_topics" folder.
2. Place the dataset in the 'data/PACS' folder making sure that the images are organized in this way:
```
data/PACS/kfold/art_painting/dog/pic_001.jpg
data/PACS/kfold/art_painting/dog/pic_002.jpg
data/PACS/kfold/art_painting/dog/pic_003.jpg
...
```

At this point you should be able to run and edit the base code provided.

## Base Code Structure

## Baseline Results (see point 0. of the project)
|          | Art Painting &#8594; Cartoon | Art Painting &#8594; Sketch | Art Painting &#8594; Photo | Average |
| :------: | :--------------------------: | :-------------------------: | :------------------------: | :-----: |
| Baseline |            54.52             |             40.44           |            95.93           |  63.63  |


## Bug Reporting
# Interactive Scene Segmentation
Interactive scene segmentation tool using Segment Anything

## Getting Started

### Prerequisites

- A graphic card that is compatible with cuda, see [compatible graphic cards](https://developer.nvidia.com/cuda-gpus)
- Anaconda, see [install Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)
- Nvidia graphic drivers, see [install Nvidia Graphic Drivers](https://wiki.ubuntuusers.de/Grafikkarten/Nvidia/nvidia/)

### Set Up

- Download the repository
- Set up a virtual conda environment with the iss.yaml file:
    ``conda env create -f iss.yaml``
- Download a pretrained model from Segment Anything, [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

    Or its lightweight version: [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Move the model into the directory `trained_models`

## Usage

Run the code by providing a path to the image you want to segment:
``python main.py --dir directory_with_training_images``

### Controls
- `1/2/3`->  to change the interaction type to "point"/ "area" / "polygon"
- `w/s` -> to segment finer/coarser
- `d`-> to save the image and get to the next one
- `q`-> to quit the program

## References

<a id="1">[1]</a> 
Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Dollär, Piotr and Girshick, Ross (2023). 
Segment Anything. 
arXiv:2304.02643.
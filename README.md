# Sequential Spatial Graphs for Video Anomaly Detection
Big Data Research Project - CentraleSupélec 2022/23

Authors: Niccolò Morabito and Yi Wu

## Data
3 video anomaly detection benchmarks have been used:
* [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
* [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)
* [StreetScene](https://www.merl.com/demos/video-anomaly-detection)

For each of them, we built:
* Yolo-annotated txt files in `data/yolo_annotated_datasets/`;
* training set pickle files which only contains normal videos in `data/training_graphs/`;
* video parameters pickle files with the video information (like width and height of the frame) in `data/video_parameters/`;
* testset pickle files which contains both normal and abnormal videos in `data/testing_graphs/`;
* testset labels pickle files in `data/testing_labels/`.

For the complete `data/` folder, please check the following [GoogleDrive link](https://drive.google.com/drive/folders/12bJFgATCoQJkGjBAQFh8F1aDIxgK-X69?usp=share_link).

## Code
The project is vided into the following folders:
* `src/grah_generation/` for the generation of NetworkX graphs starting from Yolo-annotated datasets files;
* `src/anomaly_generation/` for the corruption of graphs;
* `src/embedding_training/` for GCN and transformers;
* `src/common_utils/` for other useful code.

In order to train the model, it is sufficient to run the code contained in theh [`src/pipeline.ipynb`](src/pipeline.ipynb) Jupyter notebook.
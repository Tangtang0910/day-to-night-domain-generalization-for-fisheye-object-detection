# day-to-night-domain-generalization-for-fisheye-object-detection

annotations_creators: []
language: en
size_categories:
- 1K<n<10K
task_categories:
- object-detection
task_ids: []
pretty_name: fisheye8k
tags:
- fiftyone
- image
- object-detection
dataset_summary: '




  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 8,000 samples. The train set consists of 5,288 samples and the test set of 2,712, as marked by FiftyOne sample tags.


  ## Installation


  If you haven''t already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub


  # Load the dataset

  # Note: other available arguments include ''max_samples'', etc

  dataset = load_from_hub("Voxel51/fisheye8k")


  # Launch the App

  session = fo.launch_app(dataset)

  ```

---

# Dataset Card for FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 8000 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/fisheye8k")

# Launch the App
session = fo.launch_app(dataset)
```

The dataset is also included in the Mcity Data Engine: Mcity Data Engine: https://arxiv.org/abs/2504.21614

## Dataset Details

Fisheye lens provides omnidirectional, wide coverage for using fewer cameras to monitor road intersections, however, with view distortions.

The FishEye8K benchmark dataset for road object detection tasks comprises 157K bounding boxes across five classes (Pedestrian, Bike, Car, Bus, and Truck).

The dataset comprises 8,000 images recorded in 22 videos using 18 fisheye cameras for traffic monitoring in Hsinchu, Taiwan, at resolutions of 1080×1080 and 1280×1280.

- **Curated by:** Munkhjargal Gochoo, et. al.
- **Shared by:** [Harpreet Sahota](https://huggingface.co/harpreetsahota)
- **Language(s) (NLP):** en
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0

### Dataset Sources

- **Repository:** https://github.com/MoyoG/FishEye8K
- **Website:** https://scidm.nchc.org.tw/en/dataset/fisheye8k
- **Paper:** https://arxiv.org/abs/2305.17449

## Uses

Object detection

## Dataset Structure

  The filename is expected to follow the format: cameraX_T_YYY.png, where:
  - X is the camera number.
  - T is the time of day indicator (A for afternoon, E for evening, N for night, M for morning).
  - YYY is an arbitrary sequence of digits.

Each FiftyOne sample has two dedicated metadata fields storing the information provided in the filename, **location** for X and **time_of_day** for T. The extraction script was provided by [Daniel Bogdoll](https://github.com/daniel-bogdoll).

## Dataset Creation

### Source Data

The authors acquired 35 fisheye videos captured using 20 traffic surveillance cameras at 60 FPS in Hsinchu City, Taiwan. Among them, the first set of 30 videos (Set 1) was recorded by the cameras mounted at Nanching Hwy Road on July 17, 2018, with 1920 × 1080 resolution, and
each video lasts about 50-60 minutes. 

The second set of 5 videos (Set 2) was recorded at 1920 × 1920 resolution; each video lasts about 20 minutes.

All cameras are the property of the local police department, so there is no user consent or license issue.

All images in the dataset are made available to the public for academic and R&D use.

### Annotation Rule

The road participants were annotated based on their clarity and recognizability to the annotators,
regardless of their location. In some cases, distant objects were also annotated based on this criterion.

### Annotation

Two researchers/annotators manually labelled over 10,000 frames using the DarkLabel annotation program over one year. 

After cleaning the dataset, 8,000 frames containing 157012 bounding boxes remained. 

Unsuitable frames were removed, including those featuring road participants outside the five classes of interest

# Data Anonymization

The identification of road participants, such as people’s faces and vehicle license plates from the dataset images was found to be unfeasible due for various reasons. The cameras used for capturing the images were installed at a higher
ground level, making capturing clear facial features or license plates difficult, especially when they are far away.

Additionally, the pedestrians are not looking at the cameras, and license plates appear too small when viewed from a distance. 

However, to maintain ethical compliance and protect the privacy of the road participants, we blurred the areas of the images containing the faces of pedestrians and the
license plates of vehicles whenever they were visible.


## Citation

```bibtex
@InProceedings{Gochoo_2023_CVPR,
    author    = {Gochoo, Munkhjargal and Otgonbold, Munkh-Erdene and Ganbold, Erkhembayar and Hsieh, Jun-Wei and Chang, Ming-Ching and Chen, Ping-Yang and Dorj, Byambaa and Al Jassmi, Hamad and Batnasan, Ganzorig and Alnajjar, Fady and Abduljabbar, Mohammed and Lin, Fang-Pang},
    title     = {FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {5304-5312}
}
```

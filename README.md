![MotionMap](https://github.com/vita-epfl/MotionMap/blob/main/motionmap.png)


# MotionMap: Representing Multimodality in Human Pose Forecasting


<a href="https://arxiv.org/pdf/2412.18883"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.18883-%23B31B1B?logo=arxiv&logoColor=white" style="width: auto; height: 25px;"></a>
<a href="https://vita-epfl.github.io/MotionMap"><img alt="Project" src="https://img.shields.io/badge/Project-Page-E1AD01?style=flat&logo=googlechrome&logoColor=white&logoSize=auto&labelColor=E1AD01&color=E1AD01" style="width: auto; height: 25px;"></a>
<a href="https://hub.docker.com/repository/docker/meghshukla/motionmap/"><img alt="Docker" src="https://img.shields.io/badge/Image-motionmap-%232496ED?logo=docker&logoColor=white" style="width: auto; height: 25px;"></a>
<br>
Authors: Reyhaneh Hosseininejad* and Megh Shukla*, Saeed Saadatnejad, Mathieu Salzmann, Alexandre Alahi <br><br>

Code repository for "MotionMap: Representing Multimodality in Human Pose Forecasting", CVPR 2025. *We propose a new representation for learning multimodality in human pose forecasting which does not depend on generative models.* <br><br>

**MotionMap Saved Checkpoints: [https://drive.switch.ch/index.php/s/y9w13AnwSKy1rQe](https://drive.switch.ch/index.php/s/y9w13AnwSKy1rQe)** <br><br>



🌟 𝐌𝐨𝐭𝐢𝐨𝐧𝐌𝐚𝐩: 𝐑𝐞𝐩𝐫𝐞𝐬𝐞𝐧𝐭𝐢𝐧𝐠 𝐌𝐮𝐥𝐭𝐢𝐦𝐨𝐝𝐚𝐥𝐢𝐭𝐲 𝐢𝐧 𝐇𝐮𝐦𝐚𝐧 𝐏𝐨𝐬𝐞 𝐅𝐨𝐫𝐞𝐜𝐚𝐬𝐭𝐢𝐧𝐠 is the result of our curiosity: is diffusion for X, the current trend for solving any task, the only way forward? <br>

🚶‍♂️‍➡️ Take for instance 𝐡𝐮𝐦𝐚𝐧 𝐩𝐨𝐬𝐞 𝐟𝐨𝐫𝐞𝐜𝐚𝐬𝐭𝐢𝐧𝐠, where different future motions of a person have long been simulated using generative models like Diffusion / VAEs / GANs. However, these models rely on repeatedly sampling a large number of times to generate multimodal futures. <br>
1️⃣ This is highly inefficient, since it is hard to estimate how many samples are needed to capture the likeliest modes.<br>
2️⃣ Moreover, which of the predicted futures is the likeliest future?<br>

💡 Enter MotionMap, our novel 𝐫𝐞𝐩𝐫𝐞𝐬𝐞𝐧𝐭𝐚𝐭𝐢𝐨𝐧 𝐟𝐨𝐫 𝐦𝐮𝐥𝐭𝐢𝐦𝐨𝐝𝐚𝐥𝐢𝐭𝐲. Our idea is simple, we extend heatmaps to represent a spatial distribution over the space of motions, where different maxima correspond to different forecasts for a given observation. <br>
1️⃣ MotionMap thus allows us to represent a variable number of modes per observation and provide confidence measures for different modes. <br>
2️⃣ Further, MotionMap allows us to introduce the notion of uncertainty and controllability over the forecasted pose sequence. <br>
3️⃣ Finally, MotionMap explicitly captures rare modes that are non-trivial to evaluate yet critical for safety. <br>

📈 Our results on popular human pose forecasting benchmarks show that using a heatmap and codebook can outperform diffusion, while having multiple advantages


## Table of contents
1. [Installation: Docker](#installation)
2. [Organization](#organization)
3. [Code Execution](#execution)
4. [Acknowledgement](#acknowledgement)
5. [Citation](#citation)


## Installation: Docker <a name="installation"></a>

We provide a Docker image which is pre-installed with all required packages. We recommend using this image to ensure reproducibility of our results. Using this image requires setting up Docker on Ubuntu: [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods). Once installed, we can use the provided `docker-compose.yaml` file to start our environment with the following command:  `docker compose run --rm motionmap` <br>


## Organization <a name="organization"></a>

Running `python main.py` in the `code` folder executes the code, with configurations specified in `configuration.yml`. The method has two main stages: autoencoder training and MotionMap training. This is followed by a fine-tuning stage. The outcome of running the code across all stages will be two models: `autoencoder.pt` and `motionmap.pt`. The `code` folder contains the following files:
1. `main.py`: Main file to run the code
2. `configuration.yml`: Configuration file for the code
3. `dataset.py`: Data loading and processing for the different stages
4. `config.py`: Configuration parser
5. `autoencoder.py`: Autoencoder model training, visualization and evaluation
6. `motionmap.py`: MotionMap model training and evaluation. Visualization code is included in training and evaluation.
7. `multimodal.py`: This file integrates trained MotionMap and Autoencoder models for fine-tuning, visualization and evaluation.
8. `dataloaders.py`: Data loaders for Human3.6M and AMASS.
9. `visualizer.py`: Visualization code, specifically plotting PNG and GIFs for pose sequences.
10. `utilities.py`: Utility functions for the code
11. `metrics.py`: Evaluation metrics: Diversity, ADE, FDE, MMADE, MMFDE.

In addition, we use helper functions from ```BeLFusion```, which assist in loading the dataset and contain model definitions. The ```model``` folder contains models uniquely defined for the project such as MotionMap model (based on simple heatmaps) and Uncertainty Estimation (simple MLPs).


## Code Execution <a name="execution"></a>
We first need to activate the environment. This requires us to start the container: `docker compose run --rm motionmap`, which loads our image containing all the pre-installed packages.

The main file to run the experiment is: `main.py`. Experiments can be run using `python main.py`. The configuration file `configuration.yml` contains all the parameters for the experiment.

Stopping a container once the code execution is complete can be done using:
1. `docker ps`: List running containers
2. `docker stop <container id>`
We recommend reading the documentation on Docker for more information on managing containers.

## Acknowledgement <a name="acknowledgement"></a>

We thank `https://github.com/BarqueroGerman/BeLFusion` which contains the official implementation of BeLFusion. We use their code as a starting point. We also thank Valentin Perret, Yang Gao, Yasamin Borhani and Muhammad Osama for their valuable feedback and discussions. Finally, we are grateful to the computing team, RCP, at EPFL for their support. <br>

This This research is funded by the Swiss National Science Foundation (SNSF) through the project Narratives from the Long Tail: Transforming Access to Audiovisual Archives (Grant: CRSII5 198632). The project description is available on [https://www.futurecinema.live/project/](https://www.futurecinema.live/project/).

## Citation <a name="citation"></a>

If you find this work useful, please consider starring this repository and citing this work!

```
@InProceedings{hosseininejad2025motionmap,
  title = {MotionMap: Representing Multimodality in Human Pose Forecasting},
  author = {Hosseininejad, Reyhaneh and Shukla, Megh and Saadatnejad, Saeed and Salzmann, Mathieu and Alahi, Alexandre},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
  publisher = {IEEE/CVF}
}
```


## Miscellaneous

*Please refer to the **Controllability_Demo** branch for a simple visualization on the controllability experiments!*

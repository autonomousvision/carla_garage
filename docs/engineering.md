# Engineering

This file contains documentation about the engineering techniques we used for this project.

## Code Style
This repository follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Adherence is automatically checked by analysing the code with [Pylint](https://github.com/PyCQA/pylint). The rules are documented in [pylintrc](../pylintrc). The only modification we made was to set `max-line-length=120`. You can check the code by running `pylint --rcfile path/to/carla_garage/pylintrc` but in practice we use a PyCharm plugin. Using a coding style guide improves code consistency and readability. It requires an initial cleanup effort when applied to an existing codebase but takes very little effort to follow afterward.

We use automatic code formatting to format our code. This produces ok results and save you a lot of time thinking about how to make your code pretty. Adherence to the styleguide is also automatically enforced. We used the [Yapf](https://github.com/google/yapf) auto-formatter in our project. Its configuration style can be found in [.style.yapf](../.style.yapf). Example usage: `/path/to/yapf /path/to/carla_garage/team_code/ -i --style /path/to/carla_garage/.style.yapf -p --recursive`. Again, integrate this into your IDE so that you can run it with 1 click.

## Model Configuration Management

How to optimally manage all the hyperparameters in complex models is still an open discussion. We are using a solution here that works with argparse and pickle and a standard python class. Its advantages are that it does not have any external dependencies and handles backwards compatibility nicely. The downside is that pickle files are known to be unsafe, you should only load config files from trusted sources. This is not really an issue since you will typically only load your own configs. <br>
The general idea is that all hyperparameters of the system are stored as properties inside a [python class](../team_code/config.py). The python file contains default values for every parameter that is set during initialization of the config instance. Every parameter that can be changed as a corresponding [argparse](../team_code/train.py#L55) entry with the same name. The argparse default will be set to the config default. Adding a new configurable parameter is a simple as adding a new variable to the config class and creating the argparse argument. When training a new model, the user sets the variables he wants different from the default via the command line. After the argparse arguments are read, all parameters of the config instance will be [automatically updated](../team_code/train.py#L380) with the new values. The config instance used for the training run is then stored alongside the model as a [config.pickle file](../team_code/train.py#L583). During inference, the config instance corresponding to the model weights will be loaded, [automatically overwriting](../team_code/sensor_agent.py#L71) changed default values. If the model you are loading was trained with an older code version, it is not a problem, because new variables will be loaded with the default value. Default values should be set, so that they keep backwards compatibility. The argparse parameters are also logged as a [txt file](../team_code/train.py#L581), so one can quickly check them from an explorer. We use integers (0,1) instead of false, true because argparse does not properly support bool types.

## Flexible backbone support
Our repository supports many different backbone architecture by using the pytorch image models (TIMM) library, which is a model zoo for image classification architectures. TIMM doesn't support adding features to intermediate layers (what TransFuser does) out of the box. We write the forwards pass ourselves, mirroring TIMMs internal structure. We tested the resnet and regnet model families, but the code in principle supports any classification architecture with a 4 block structure. The backbones can be selected by setting the training options `--image_architecture` and `--lidar_architecture` to the corresponding TIMM model name (e.g. `resnet34` or `regnety_032`).
A great thing about TIMM is that it does not have cuda dependencies, which save users many cuda compatibility troubles when setting up their machines. We have removed dependencies on other model zoos for that reason.

## Coordinate Systems
Different data sources lie in different coordinate systems, which can lead to confusion and bugs when processing them. To address this we convert all data into the same unified coordinate system upon arrival. More documentation on the coordinate systems can be found [here](coordinate_systems.md).

## Localization
The GPS sensor has strong Gaussian noise applied to it in the CARLA leaderboard setting.
During data collection, we use ground truth localization instead to have clean data.
During inference, one can obtain a better signal by applying a nonlinear filter together with a vehicle model.
We chose the Unscented Kalman Filter (UKF) technique paired with a kinematic bicycle model.
Such filter techniques improve localization by predicting once own motion with a model and comparing the estimation with the next sensor observation.
The kinematic bicycle model parameters were taken from World on Rails, where they were optimized with gradient descent to match the standard CARLA vehicle.
We estimated the parameters of the UKF in a similar fashion by collecting a small validation dataset of GPS measurements and ground truth vehicle locations. Parameters were then manually tuned to minimize mean squared error between the filter prediction and the ground truth. Framing filtering as a supervised learning problem is maybe a bit unusual but worked quite well for our problem.

## Ensembling
The inference code automatically loads all model files in the provided config folder. It forwards passes all models sequentially and averages the resulting predictions. Bounding box predictions are combined using non-maximum suppression. 

## Inference options
Models have parameters that can be changed during different inference runs.
Unlike the fixed training parameters they are set via environment variables.
The available option are:
```Shell
Sensor / Map agent:
DIRECT=1 # 0: Use waypoints as output representation, 1: Use path + target speed as output representation
UNCERTAINTY_WEIGHT=1 # Used with direct=1, Whether to use an argmax target speed, or weight the target speed by the predicted probabilities.
UNCERTAINTY_THRESHOLD=0.5 # Used with direct=1, Probability of the 0 class at which a full brake will be used
SLOWER=1 # Used with direct=1, Target speed will be reduced by 1 m/s compared to the dataset.
SAVE_PATH=None # If set to system folder, this folder will be used as route to store logging and debug information.
DEBUG_CHALLENGE=0 # 1: Generate visualization images at SAVE_PATH
TP_STATS=0 # 1: Collects some statistics about the target point
STOP_CONTROL=1 # 1: Clear stop signs detected in the object detector by setting the speed to 0 in the controller.
HISTOGRAM=0 # 1: Collect target speeds extracted by the controller. Stored at SAVE_PATH

Data collection:
DATAGEN=0 # 1: Generates and stores data at SAVE_PATH. Also sets evaluation seed to random.
TMP_VISU=0 # 1: Store BEV semantic segmentation as RGB on disk

Plant
VISU_PLANT=0 # 1: Generate visualization images at SAVE_PATH for the plant model
```
Additionally, there is the evaluation parameter `BLOCKED_THRESHOLD=180`.
It determines, after how many seconds, stopping the agent will incur stop infractions. We make this a parameter here because it was [silently changed](https://github.com/carla-simulator/leaderboard/commit/bd9e75500c9c20b45a0609c701387a96492bd60f) at the end of 2020 by the CARLA leaderboard team from 90 → 180. This changes the evaluation significantly because the simulator de-spawns other cars if they stand still for more than 90 seconds. It can sometimes occur that the ego vehicle and another car are blocking each other's path. With a value of 180 the other blocking car will be de-spawned and the situation does not require a backwards maneuver to be solved. With 90 seconds, the ego vehicle will incur a block infraction instead (if it does not attempt to clear the other vehicle's path). All of our experiments used the newer value of 180. It is unclear which value is used on the leaderboard severs.

## Compression
Our dataset consists of diverse multi-modal labels and sensor data. Storing all this data, particularly with at higher scales, is demanding in terms of disk space. We use strong compression algorithms to counteract that. Images are JPG compressed. This is a much stronger compression than lossless PNG but introduces artifacts into the image. To avoid a distribution shift at test time, we simply use in memory JPG compression and decompression during inference as well. As TXT files are compressed with GZIP. Semantic segmentation and depth are stored as PNG files. Lossless compression is needed here because artifacts could change the labels. PNG compression is not very strong with depth maps, particularly at high precision. We store them at [8 bit](../team_code/data_agent.py#L185) for that reason, at some loss of precision (~20 cm). This is not important for us, as depth maps are only auxiliary tasks. If one cares about depth prediction, consider changing this to 16 bit, where the resolution loss becomes negligible.
We use a specialized algorithm called laszip to compress our [LiDAR point clouds](../team_code/data_agent.py#L359). The algorithm achieves ~5x stronger compression than any other generic compression algorithm that we tested at no effective loss in resolution.
Overall, this makes the size of our dataset quite manageable (< 500 GB). What could still be improved is the number of files, which are currently > 10M. This is because every sample has lots of labels and sensor files. Some file systems are inefficient at handling many small files. This could potentially be improved by merging multiple files into 1 on disk (or using some sort of database system) as they are typically loaded together anyway.
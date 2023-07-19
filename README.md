<p align="center">
  <img src="assets/carla_garage_white.png" alt="CARLA garage" width="500"/>
  <h3 align="center">
        <a href="https://arxiv.org/abs/2306.07957.pdf"> Paper</a> | <a href="https://youtu.be/ChrPW8RdqQU">Video</a> | <a href="https://youtu.be/x_42Fji1Z2M?t=1073">Talk</a> | <a href="https://www.cvlibs.net/shared/common_misconceptions.pdf"> Slides</a>
  </h3>
</p>
    


> [**Hidden Biases of End-to-End Driving Models**](https://arxiv.org/abs/2306.07957) <br>
> [Bernhard Jaeger](https://kait0.github.io/), [Kashyap Chitta](https://kashyap7x.github.io/), [Andreas Geiger](https://www.cvlibs.net/) <br>
> International Conference on Computer Vision (ICCV), 2023
> 
> This repo contains the code for the paper [**Hidden Biases of End-to-End Driving Models**](https://arxiv.org/abs/2306.07957) . \
> We provide clean, configurable code with documentation as well as pre-trained weights with strong performance. \
> The repository can serve as a good starting point for end-to-end autonomous driving research on [CARLA](https://github.com/carla-simulator/carla).

## Contents

1. [Setup](#setup)
2. [Pre-Trained Models](#pre-trained-models)
3. [Evaluation](#evaluation)
4. [Dataset](#dataset)
4. [Data generation](#data-generation)
5. [Training](#training)
6. [Additional Documenation](#additional-documentation)
7. [Citation](#citation)

## Setup

Clone the repo, setup CARLA 0.9.10.1, and build the conda environment:

```Shell
git clone https://github.com/autonomousvision/carla_garage.git
cd carla_garage
chmod +x setup_carla.sh
./setup_carla.sh
conda env create -f environment.yml
conda activate garage
```
Before running the code, you will need to add the following paths to PYTHONPATH on your system:
```Shell
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```
You can add this in your shell scripts or directly integrate it into your favorite IDE. \
E.g. in PyCharm: Settings -> Project -> Python Interpreter -> Show all -> garage (need to add from existing conda environment first) -> Show Interpreter Paths -> add all the absolute paths above (without pythonpath).

## Pre-Trained Models
We provide a set of pretrained models [here](https://s3.eu-central-1.amazonaws.com/avg-projects-2/jaeger2023arxiv/models/pretrained_models.zip).
The models are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
These are the final model weights used in the paper, the folder indicates the benchmark.
For the training and validation towns, we provide 3 models which correspond to 3 different training seeds. The format is `approach_trainingsetting_seed`. Each folder has an `args.txt` containing the training settings in text, a `config.pickle` containing all hyperparameters for the code and a `model_0030.pth` containing the model weights. Additionally, there are training logs for most models.

## Evaluation

To evaluate a model, you need to start a CARLA server:
```Shell
cd /path/to/CARLA/root
./CarlaUE4.sh -opengl
```
Afterward, run [leaderboard_evaluator_local.py](leaderboard/leaderboard/leaderboard_evaluator_local.py) as the main python file.
It is a modified version of the original leaderboard_evaluator.py which has the configurations used in the benchmarks we consider and additionally provides extra logging functionality.

Set the `--agent-config` option to a folder containing a `config.pickle` and `model_0030.pth`. <br>
Set the `--agent` to [sensor_agent.py](team_code/sensor_agent.py). <br>
The `--routes` option should be set to [lav.xml](leaderboard/data/lav.xml) or [longest6.xml](leaderboard/data/longest6.xml). <br>
The `--scenarios ` option should be set to [eval_scenarios.json](leaderboard/data/scenarios/eval_scenarios.json) for both benchmarks. <br>
Set `--checkpoint ` to `/path/to/results/result.json`

To evaluate on a benchmark, set the respective environment variable: `export BENCHMARK=lav` or `export BENCHMARK=longest6`. <br>
Set `export SAVE_PATH=/path/to/results` to save additional logs or visualizations

Models have inference options that can be set via environment variables.
For the longest6 model you need to set `export UNCERTAINTY_THRESHOLD=0.33`, for the LAV model `export STOP_CONTROL=1` and for the leaderboard model `export DIRECT=0`.
Other options are correctly set by default. <br>
For an example, you can check out [local_evaluation.sh](leaderboard/scripts/local_evaluation.sh).

After running the evaluation, you need to parse the results file with [result_parser.py](tools/result_parser.py).
It will recompute the metrics (the initial once are [incorrect](https://github.com/carla-simulator/leaderboard/issues/117)) compute additional statistics and optionally visualize infractions as short video clips.
```Shell
python ${WORK_DIR}/tools/result_parser.py --xml ${WORK_DIR}/leaderboard/data/lav.xml --results /path/to/results --log_dir /path/to/results
```

The result parser can optionally create short video/gif clips showcasing re-renderings of infractions that happened during evaluation. The code was developed by Luis Winckelmann and Alexander Braun.
To use this feature, you need to prepare some map files once (that are too large to upload to GitHub). For that, start a CARLA sever on your computer and run [prepare_map_data.py](tools/proxy_simulator/prepare_map_data.py).
Afterward, you can run the feature by using the `--visualize_infractions` flag in [result_parser.py](tools/result_parser.py). The feature requires logs to be available in your results folder, so you need to set `export SAVE_PATH=/path/to/results` during evaluation.

### How to actually evaluate
The instructions above are what you will be using to debug the code. Actually evaluating challenging benchmarks such as longest6, that have over 108 long routes, is very slow in practice. Luckily, CARLA evaluations are [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel). Each of the 108 routes can be evaluated independently in parallel. That means if you have 2 GPUs you can evaluate 2x faster, if you have 108 GPUs you can evaluate 108x faster. While using the same amount of overall compute. To do that, you need access to a scalable cluster system and some scripts to parallelize. We are using [SLURM](https://slurm.schedmd.com/overview.html) at our institute. To evaluate a model, we are using the script [evaluate_routes_slurm.py](evaluate_routes_slurm.py). It is intended to be run inside a tmux on an interactive node and will spawn evaluation jobs (up till the number set in [max_num_jobs.txt](max_num_jobs.txt)). It also monitors the jobs and resubmits jobs where it detected a crash. In the end, the script will run the result parser to aggregate the results. If you are using a different system, you can use this as guidance and write your own script. The CARLA leaderboard benchmarks are the most challenging in the driving scene right now, but if you don't have access to multiple GPUs you might want to use simulators that are less compute intensive for your research. NuPlan is a good option, and our group also provides [strong baselines for nuPlan](https://github.com/autonomousvision/nuplan_garage).

## Dataset
We released the dataset we used to train our final models.
The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
You can download it using:
```Shell
cd /path/to/carla_garage/tools
bash download_data.sh
```
The script will download the data to `/path/to/carla_garage/data`. This is also the path you need to set `--root_dir` to for training. The script will download and unzip the data with 11 parallel processes. The download is roughly 350 GB large (will be a bit more after unzipping).

## Data Generation
Dataset generation is similar to evaluation. You can generate a dataset by changing the `--agent` option to [data_agent.py](team_code/data_agent.py) and the `--track` option to `MAP`. In addition, you need to set the following environment flags:
```Shell
export COMPLETION_PERCENT=85
export DATAGEN=1
export BENCHMARK=collection
export CHECKPOINT_ENDPOINT=/path/to/dataset/Routes_{route}_Repetition{repetition}/Dataset_generation_{route}_Repetition{repetition}.json
export SAVE_PATH=/path/to/dataset/Routes_{route}_Repetition{repetition}
```
Again it is too slow to generate our dataset with a single computer, you should be using multiple GPUs. We provide a [python script](generate_dataset_slurm.py) for SLURM clusters, it works in the same fashion as the evaluation script.
We will release the dataset we used at a later point.

## Training
Agents are trained via the file [train.py](team_code/train.py).
Examples how to use it are provided for [shell](team_code/shell_train.sh) and [SLURM](team_code/slurm_train.sh).
You need to activate garage conda environment before running it.
It first sets the relevant environment variables and then launches the training with torchrun.
Torchrun is a pytorch tool that handles multi-gpu training.
If you want to debug on a single gpu simply set `--nproc_per_node=1`.
The training script has many options to configure your training you can list them with `python train.py --help` or look through the code.
The most important once are: 
```Shell
--id your_model_000 # Name of your experiment
--batch_size 32 # Batch size per GPU
--setting all # Which towns to withhold during training. Use 'all' for leaderboard, longest6 and '02_05_withheld' for LAV models.
--root_dir /path/to/dataset # Path to the root_dir of your dataset
--logdir /path/to/models # Root dir where the training files will be stored
--use_controller_input_prediction 1 # Whether your model trains with a classification + path prediction head
--use_wp_gru 0 # Whether you model trains with a waypoint head.
--use_discrete_command 1 # Whether to use the navigational command as input to the model
--use_tp 1  # Whether to use the target point as input to your model
--cpu_cores 20 # Total number of cpu cores on your machine
--num_repetitions 3 # How much data to train on (Options are 1,2,3). 1x corresponds to 185k in Table 5, 3x corresponds to 555k
```
Additionally, to do the two stage training from Table 4 you need the `--continue_epoch` and `--load_file` option.
You need to train twice. First train a model with set `--use_wp_gru 0` and `--use_controller_input_prediction 0`, this will only train the perception backbone with auxiliary losses.
Then, train a second model, set e.g. `--use_controller_input_prediction 1`, `--continue_epoch 0` and `--load_file /path/to/stage1/model_0030.pth`.
The `load_file` option is usually used to resume a crashed training, but with `--continue_epoch 0` the training will start from scratch with the pre-trained weights used for initialization.

The training dataset will be released at a later point.

### Training in PyCharm
You can also run and debug torchrun in PyCharm. To do that you need to set your run/debug configuration as follows:\
Set the script path to: `/path/to/train.py` \
Set the interpreter options to:
```Shell
-m torch.distributed.run --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d
```
Training parameters should be set in the `Parameters:` field and environment variable in `Environment Variables:`.
Additionally, you need to set up conda environment (variables) as described above.

## Submitting to the CARLA leaderboard
To submit to the CARLA leaderboard, you need docker installed on your system (as well as the nvidia-container-toolkit to test it). Create the folder `team_code/model_ckpt/transfuser`. Copy the model.pth files and config.pickle that you want to evaluate to team_code/model_ckpt/transfuser. If you want to evaluate an ensemble, simply copy multiple .pth files into the folder, the code will load all of them and ensemble the predictions.
Edit the environment paths at the top of `tools/make_docker.sh` and then:

```Shell
cd tools
./make_docker.sh
```

The script will create a docker image with the name transfuser-agent.
Before submitting, you should locally test your image. To do that, start up a CARLA server on your computer (it will be able to communicate with the docker container via ports). Then start your docker container. An example is provided in [run_docker.sh](tools/run_docker.sh).
Inside the docker container start your agent using:
```Shell
cd leaderboard
cd scripts
bash run_evaluation.sh
```
You can stop the evaluation, after confirming that there is no issue, using "ctrl + c + \". <br>
To submit, follow the instructions on the [leaderboard](https://leaderboard.carla.org/submit_v1/) to make an account and install alpha.

```Shell
alpha login
alpha benchmark:submit  --split 3 transfuser-agent:latest
```
The command will upload the docker image to the cloud and evaluate it.

## Additional Documentation
- **Coordinate systems** in CARLA repositories are usually a big mess. In this project, we addressed this by changing all data into a unified coordinate frame. Further information about the coordinate system can be found [here](docs/coordinate_systems.md).

- The TransFuser model family has grown quite a lot with different variants, which can be confusing for new community members. The **[history](docs/history.md)** file explains the different versions and which paper you should cite to refer to them.

- Building a full autonomous driving stack involves quite some [**engineering**](docs/engineering.md). The documentation explains some of the techniques and design philosophies we used in this project.

- The codebase can run any experiment presented in the paper. It also supports some additional features that we did not end up using. They are documented [here](docs/additional_features.md).

## Contact
If you have any questions or suggestions, please feel free to open an issue or contact us at bernhard.jaeger@uni-tuebingen.de.

## Citation
If you find CARLA garage useful, please consider giving us a star &#127775; and citing our paper with the following BibTeX entry.

```BibTeX
@article{Jaeger2023ICCV,
  title={Hidden Biases of End-to-End Driving Models},
  author={Jaeger, Bernhard and Chitta, Kashyap and Geiger, Andreas},
  booktitle={Proc. of the IEEE International Conf. on Computer Vision (ICCV)},
  year={2023}
}
```


## Acknowledgements
Open source code like this is build on the shoulders of many other open source repositories.
In particularly we would like to thank the following repositories for their contributions:
* [simple_bev](https://github.com/aharley/simple_bev)
* [transfuser](https://github.com/autonomousvision/transfuser)
* [interfuser](https://github.com/opendilab/InterFuser)
* [mmdet](https://github.com/open-mmlab/mmdetection)
* [roach](https://github.com/zhejz/carla-roach/)
* [plant](https://github.com/autonomousvision/plant)
* [king](https://github.com/autonomousvision/king)
* [wor](https://github.com/dotchen/WorldOnRails)
* [tcp](https://github.com/OpenDriveLab/TCP)
* [lbc](https://github.com/dotchen/LearningByCheating)

We also thank the creators of the numerous pip libraries we use. Complex projects like this would not be feasible without your contribution.

<p align="center">
  <img src="assets/carla_garage_white.png" alt="CARLA garage" width="500"/>
  <h3 align="center">
        <a href="https://arxiv.org/pdf/2306.07957.pdf"> PDF</a> | Video | Talk | <a href="https://www.cvlibs.net/shared/common_misconceptions.pdf"> Slides</a>
  </h3>
</p>
    


> [**Hidden Biases of End-to-End Driving Models**](https://arxiv.org/abs/2306.07957)
>
> [Bernhard Jaeger](https://kait0.github.io/), [Kashyap Chitta](https://kashyap7x.github.io/), and [Andreas Geiger](https://www.cvlibs.net/)
> 
> University of Tübingen, Tübingen AI Center
> 
> This repo contains the code for the paper [**Hidden Biases of End-to-End Driving Models**](https://arxiv.org/abs/2306.07957) . \
> We provide clean, configurable code with documentation as well as pre-trained weights with strong performance. \
> The repository can serve as a good starting point for end-to-end autonomous driving research on [CARLA](https://github.com/carla-simulator/carla).

## To Do
- [ ] Dataset
- [ ] Documentation
- [x] Pre-trained models
- [x] Training code
- [x] Data collection code
- [x] Additional scripts
- [x] Inference code
- [x] Initial repo & paper

## Contents

1. [Setup](#setup)
2. [Pre-Trained Models](#pre-trained-models)
3. [Evaluation](#evaluation)

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
We provide a set of pretrained models [here](https://drive.google.com/file/d/1dDeeRrPL7UAqqHn4RMdg3DrvHSlNgzmE/view?usp=sharing).
These are the final model weights used in the paper, the folder indicates the benchmark.
For the training and validation towns, we provide 3 models which correspond to 3 different training seeds.

## Evaluation

To evaluate a model, you should run [leaderboard_evaluator_local.py](leaderboard/leaderboard/leaderboard_evaluator_local.py) as the main python file.
It is a modified version of the original leaderboard_evaluator.py which has the configurations used in the benchmarks we consider and additionally provides extra logging functionality.

Set the `--agent-config` option to a folder containing a `config.pickle` and `model_0030.pth`. <br>
Set the `--agent_file` to [sensor_agent.py](team_code/sensor_agent.py). <br>
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


## Contact
If you have any questions or suggestions, please feel free to open an issue or contact us at bernhard.jaeger@uni-tuebingen.de.


## Citation
If you find CARLA garage useful, please consider giving us a star &#127775; and citing our paper with the following BibTeX entry.

```BibTeX
@article{Jaeger2023ARXIV,
  title={Hidden Biases of End-to-End Driving Models},
  author={Jaeger, Bernhard and Chitta, Kashyap and Geiger, Andreas},
  journal={arXiv},
  volume={2306.07957},
  year={2023}
}
```


## Acknowledgements
Opensource code like this is build on the shoulders on the shoulders of many other opensource repositories.
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

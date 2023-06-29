# History

There are many different version of TransFuser by now and we sometimes see the wrong papers getting cited.
Here is a short history over the different transFuser versions and how to cite them correctly:

### TransFuser (CVPR 2021)
The first version of TransFuser. The paper [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://www.cvlibs.net/publications/Prakash2021CVPR.pdf) introduced the architecture back in 2021.
The code is still available and can be found [here](https://github.com/autonomousvision/transfuser/tree/cvpr2021).
The models were developed in the early days of the CARLA leaderboard code / community, where dataset quality was quite poor. There is not much of a point to compare to this model any more unless you want to make your life easy.
```BibTeX
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and
            Chitta, Kashyap and
            Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```

### TransFuser+ (Master Thesis University of Tübingen 2021)
The thesis [Expert Drivers for Autonomous Driving](https://kait0.github.io/assets/pdf/master_thesis_bernhard_jaeger.pdf) investigated the data quality issue of the original TransFuser work. 
It proposes a stronger automatic labeling algorithm. 
Together with auxiliary training, this pushed the performance of TransFuser a lot without changing the architecture.
The document is available online but not formally published as it is only relevant to the core CARLA community.
We would still appreciate it if you cite it where relevant.
The code and models are not directly released, but the relevant code was published as part of the PAMI project.
```BibTeX
@mastersthesis{Jaeger2021Thesis, 
	author = {Bernhard Jaeger}, 
	title = {Expert Drivers for Autonomous Driving}, 
	year = {2021}, 
	school = {University of Tübingen}, 
}
```

### TransFuser (T-PAMI 2022)
The journal update to the CVPR paper. The paper [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf) is published in Transaction on Pattern Analysis and Machine Intelligence.
At the core it is still the TransFuser architecture, but it features better data, sensors, backbones, training and a rigorous set of ablations that shows what is important and what is not.
The paper features a version called Latent TransFuser, which is a camera only TransFuser that replaces the LiDAR input by a positional encoding.
The final models are roughly 4x better on the CARLA leaderboard than the CVPR TransFuser. 
This is the model you should be comparing to when you want to claim your method outperforms TransFuser.
Code, models and data are available [online](https://github.com/autonomousvision/transfuser/).
```BibTeX
@article{Chitta2022PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2022},
}
```

### TransFuser++ (ArXiv 2023)
The pre-print [Hidden Biases of End-to-End Driving Models](https://arxiv.org/abs/2306.07957) offers some explanations why TransFuser and related approaches work so well.
It also contains the latest and best performing models in the TransFuser family, called TransFuser++ (as well as a WP variant that uses waypoints as output representation).
[Code](https://github.com/autonomousvision/carla_garage) is available online. TransFuser++ is what you need to outperform to claim state-of-the-art performance.
```BibTeX
@article{Jaeger2023ARXIV,
  title={Hidden Biases of End-to-End Driving Models},
  author={Jaeger, Bernhard and Chitta, Kashyap and Geiger, Andreas},
  journal={arXiv},
  volume={2306.07957},
  year={2023}
}
```
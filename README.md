# ViP-NeRF
Official code release accompanying the SIGGRAPH 2023 paper - ["ViP-NeRF: Visibility Prior for Sparse Input Neural Radiance Fields"](https://nagabhushansn95.github.io/publications/2023/ViP-NeRF.html). All the published data is available [here](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/ErmKnJtw5mRPpEzxzVbLMV4BH_XEc1jQDhmtqait2NCqDA?e=E16C2a).

## Python Environment
Environment details are available in `EnvironmentData/ViP_NeRF_GPU.yml`. The environment can be created using conda
```shell
conda env create -f ViP_NeRF_GPU.yml
```

## Add the source directory to PYTHONPATH
```shell
export PYTHONPATH=<ABSOLUTE_PATH_TO_VIPNERF_DIR>/src:$PYTHONPATH
```

## Set-up Databases
Please follow the instructions in [database_utils/README.md](src/database_utils/README.md) file to set up various databases. Instructions for custom databases are also included here.

## Generate Priors
### Sparse Depth Prior
Please follow the instructions in [prior_generators/sparse_depth/README.md](src/prior_generators/sparse_depth/README.md) file to generate sparse depth prior.

### Dense Visibility Prior
Please follow the instructions in [prior_generators/visibility/README.md](src/prior_generators/visibility/README.md) file to generate dense visibility prior.

## Training and Inference
The files `RealEstateTrainerTester01.py`, `NerfLlffTrainerTester01.py` and `DtuTrainerTester01.py` contain the code for training, testing and quality assessment along with the configs for the respective databases.
```shell
cd src/
python RealEstateTrainerTester01.py
python NerfLlffTrainerTester01.py
python DtuTrainerTester01.py
cd ../
```

## Inference with Pre-trained Models
The train configs are also provided in `runs/training/train****` folders for each of the scenes. Please download the trained models from [here](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/EssGwn7AUh5AjT6wtbcnsp4B7QGHQ7-DAeAiBBdTBkwilg?e=GCZSAQ) and place them in the appropriate folders. Disable the train call in the [TrainerTester](src/RealEstateTrainerTester01.py#L340) files and run the respective files. This will run inference using the pre-trained models and also evaluate the synthesized images and reports the performance. To reproduce results from the paper, use the models trained for 50k iterations. For best results, use the models trained for more iterations.

## License
MIT License

Copyright (c) 2023 Nagabhushan Somraj, Rajiv Soundararajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Citation
If you use this code for your research, please cite our paper

```bibtex
@article{somraj2023VipNeRF,
    title = {{ViP-NeRF}: Visibility Prior for Sparse Input Neural Radiance Fields},
    author = {Somraj, Nagabhushan and Soundararajan, Rajiv},
    booktitle = {ACM Special Interest Group on Computer Graphics and Interactive Techniques (SIGGRAPH)},
    month = {August},
    year = {2023},
    doi = {10.1145/3588432.3591539},
}
```
If you use outputs/results of ViP-NeRF model in your publication, please specify the version as well. The current version is 1.0.

## Acknowledgements
Our code is built on top of [NeRFs-Simplified](https://github.com/NagabhushanSN95/NeRFs-Simplified) codebase.


For any queries or bugs regarding ViP-NeRF, please raise an issue.
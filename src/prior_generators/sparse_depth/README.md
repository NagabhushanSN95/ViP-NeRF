# Sparse Depth Prior Generation

We use Colmap to generate sparse depth. Installation instructions can be found [here](https://colmap.github.io/install.html).
Run the following files to generate sparse depth priors for the respective datasets for all the three input configurations.
```shell
cd src/prior_generators/sparse_depth/
python DepthEstimator01_RealEstate.py
python DepthEstimator02_NeRF_LLFF.py
python DepthEstimator05_DTU.py
cd ../../../
```



## Acknowledgements
Parts of the code are borrowed from [DS-NeRF](https://github.com/dunbar12138/DSNeRF) codebase.
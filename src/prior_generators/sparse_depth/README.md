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

Running the above files creates a new directory `data/databases/<DATABASE_NAME>/data/all/estimated_depths`, which contains three sub-directories named `DE02,DE03,DE04` corresponding to two, three and four input-view settings. Each of these directories will contain multiple sub-directories, one for every scene in the dataset. The following tree shows an exmaple.
```
data/databases/NeRF_LLFF/data/all/estimated_depths
|--DE02
|  |--fern
|  |  |--estimated_depths_down4
|  |  |  |--0006.csv
|  |  |  |--0013.csv
|  |  |--EstimatedBounds.csv
|  |--flower
|  ...  
|--DE03
|  |--fern
|  ...
|--DE04
|  |--fern
|  ...
```

Note that Colmap can sometime fail in 3D sparse reconstruction. This is more prevalent on some DTU scenes, for e.g., the one containing a single homogeneous ball.
Re-running Colmap (above python files) can sometimes fix the issue. Even if the sparse depth prior generation fails, ViP-NeRF can still run on such scenes by utilizing the visibility prior. Simple disable sparse depth prior for such scenes. For example, comment lines 306--309 in [DtuTrainerTester01.py](../../../src/DtuTrainerTester01.py).



## Acknowledgements
Parts of the code are borrowed from [DS-NeRF](https://github.com/dunbar12138/DSNeRF) codebase.
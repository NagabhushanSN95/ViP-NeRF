# Dense Visibility Prior Generation

Run the following files to generate sparse depth priors for all the three datasets for all the three input configurations.
```shell
cd src/prior_generators/visibility/
python VisibilityMask01_RealEstate.py
python VisibilityMask02_NeRF_LLFF.py
python VisibilityMask05_DTU.py
cd ../../../
```

Running the above files creates a new directory `data/databases/<DATABASE_NAME>/data/all/visibility_prior`, which contains three sub-directories named `VW02,VW03,VW04` corresponding to two, three and four input-view settings. Each of these directories will contain multiple sub-directories, one for every scene in the dataset. The following tree shows an exmaple.
```
data/databases/NeRF_LLFF/data/all/estimated_depths
|--VW02
|  |--fern
|  |  |--visibility_masks
|  |  |  |--0006.npy
|  |  |  |--0006.png
|  |  |  |--0013.npy
|  |  |  |--0013.png
|  |  |--visibility_weights
|  |  |  |--0006.npy
|  |  |  |--0006.png
|  |  |  |--0013.npy
|  |  |  |--0013.png
|  |--flower
|  ...  
|--VW03
|  |--fern
|  ...
|--VW04
|  |--fern
|  ...
```
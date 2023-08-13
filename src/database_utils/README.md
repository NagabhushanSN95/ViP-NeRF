# Extracting Databases

## RealEstate-10K
1. Download the dataset metadata from [here](https://google.github.io/realestate10k/download.html)
2. Remap the names to integers:
```shell
python VideoNameMapper.py
```
3. We use the scenes from test set. Select the scenes to download:
```shell
python SceneSelector01.py
```
4. Download the selected scenes:
```shell
python DataExtractor01.py
```
5. Copy the downloaded scenes to `Data/databases/RealEstate10K/data/test/database_data`.
6. Train/test configs are already provided in the github repository. In case you want to create them again: 
```shell
python TrainTestCreator01.py
python VideoPoseCreator01_Original.py
```

## NeRF-LLFF
1. Download the [`nerf_llff_data.zip`](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=share_link) file from original release in google drive. Place the downloaded file at `Data/databases/NeRF_LLFF/data/nerf_llff_data.zip`.
2. Run the data extractor file:
```shell
cd src/database_utils/nerf_llff/data_organizers/
python DataExtractor01.py
```
3. Train/test configs are already provided in the github repository. In case you want to create them again: 
```shell
python TrainTestCreator01_UniformSparseSampling.py
python VideoPoseCreator01_Spiral.py
```
4. Return to root directory
```shell
cd ../../../../
```

## DTU
1. Download the dataset provided by pixelNeRF [here](https://drive.google.com/file/d/1aTSmJa8Oo2qCc2Ce2kT90MHEA6UTSBKj/view?usp=share_link).
2. Unzip the downloaded file and place the unzipped data in `data/databases/DTU/data/all/unzipped_data/PixelNeRF/` <br>
The path of `scan1` should be `data/databases/DTU/data/all/unzipped_data/PixelNeRF/rs_dtu_4/DTU/scan1`
3. Extract the data:
```shell
cd src/database_utils/dtu/data_organizers/
python DataExtractor01_PixelNeRF.py
```
4. Download the object masks data provided by RegNeRF [here](https://drive.google.com/file/d/1Yt5T3LJ9DZDiHbtd9PDFNHqJAd7wt-_E/view?usp=sharing).
5. Place the downloaded files in `data/databases/DTU/data/all/unzipped_data/RegNeRF` <br>
The path of `scan8` should be `data/databases/DTU/data/all/unzipped_data/RegNeRF/idrmasks/scan8`
6. Extract the object masks data:
```shell
python DataExtractor02_RegNeRF.py
```
7. Train/test configs are already provided in the github repository. In case you want to create them again: 
```shell
python TrainTestCreator01_PixelNeRF.py
python TrainTestCreator02_PixelNeRF.py
```
To use different train/test sets, modify the above files accordingly and regenerate the train/test configs.
8. Return to root directory
```shell
cd ../../../../
```

## Custom Databases
We use the Open CV convention: `(x, -y, -z)` world-to-camera format to store the camera poses. 
The camera intrinsics and extrinsics are stored in the `csv` format after flattening them, i.e., if a scene contains 50 frames, intrinsics and extrinsics are stores as csv files with 50 rows each and 9 & 16 columns respectively.
The directory tree in the following shows an example.
Please refer to one of the [data-loaders](../data_loaders/RealEstateDataLoader01.py) for more details. 
Organize your custom dataset in accordance with the data-loader or write the data-loader file to load the data directly from your custom database format.

Example directory tree:
```shell
<DATABASE_NAME>
 |--data
    |--all
    |  |--database_data
    |     |--scene0001
    |     |  |--rgb
    |     |  |  |--0000.png
    |     |  |  |--0001.png
    |     |  |  |-- ...
    |     |  |  |--0049.png
    |     |  |--CameraExtrinsics.csv
    |     |  |--CameraIntrinsics.csv
    |     |--scene0002
    |     | ...
    |--train_test_sets
```

Our code also requires a config file specify the train/validation/test images. Please look into [train-test-creators](real_estate_10k/train_test_creators/TrainTestCreator01.py) and replicate a similar file for your custom dataset.

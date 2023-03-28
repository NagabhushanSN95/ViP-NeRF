# Extracting Databases

## RealEstate-10K
1. Download the dataset metadata from [here](https://google.github.io/realestate10k/download.html)
2. Remap the names to integers: <br/>
    `python VideoNameMapper.py`
3. We use the scenes from test set. Select the scenes to download: <br/>
    `python SceneSelector01.py`
4. Download the selected scenes: <br/>
    `python DataExtractor01.py`
5. Copy the downloaded scenes to `Data/Databases/RealEstate10K/Data/test/DatabaseData`.
6. Create the train/test configs: <br/> 
    `python TrainTestCreator01.py` <br/>
    `python VideoPoseCreator01_Original.py` <br/>

## NeRF-LLFF
1. Download the [`nerf_llff_data.zip`](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=share_link) file from original release in google drive. Place the downloaded file at `Data/Databases/NeRF_LLFF/Data/nerf_llff_data.zip`.
2. Run the data extractor file: <br/>
    `python DataExtractor01.py`
3. Create the train/test configs: <br/> 
    `python TrainTestCreator01_UniformSparseSampling.py` <br/>
    `python VideoPoseCreator01_Spiral.py` <br/>

## DTU
1. Download the dataset provided by pixelNeRF [here](https://drive.google.com/file/d/1aTSmJa8Oo2qCc2Ce2kT90MHEA6UTSBKj/view?usp=share_link).
2. Unzip the downloaded file and place the unzipped data in `Data/Databases/DTU/Data/UnzippedData/PixelNeRF/`
3. Extract the data: <br/>
    `python DataExtractor01_PixelNeRF.py`
4. Download the object masks data provided by RegNeRF [here](https://drive.google.com/file/d/1Yt5T3LJ9DZDiHbtd9PDFNHqJAd7wt-_E/view?usp=sharing).
5. Place the downloaded files in `Data/Databases/DTU/Data/UnzippedData/RegNeRF/idrmasks`
6. Extract the object masks data: <br/>
    `python DataExtractor02_RegNeRF.py`
7. Create the train/test configs: <br/> 
    `python TrainTestCreator01_PixelNeRF.py` <br/>
    `python TrainTestCreator02_PixelNeRF.py` <br/>
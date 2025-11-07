## Latent Graph Structure Learning for Large-Scale Traffic Forecasting


<img width="782" height="316" alt="image" src="https://github.com/user-attachments/assets/6cc0bb9c-5194-4a57-aa18-d1dd0cb5c278" />


## Requirements
python 3.9.21 \
torch==1.12.1+cu113 \
numpy==1.23.0 \
pandas==2.2.3 \
tqdm==4.67.1


## Datasets
The datasets are available at https://drive.google.com/drive/folders/1nsgMDU_-0Bb3r0Jwv1W_FsgygfZLNHFc?usp=drive_link \
You can also download from https://github.com/LMissher/PatchSTG


## Usage
Download datasets and put them in the `./data` folder. Create an empty `./cpt` folder. \
Then, you can train the model as follows. Taken SD dataset for example.
```
python main.py --config ./config/SD.conf
```

## Acknowledgement
This repository uses codes from PatchSTG (https://github.com/LMissher/PatchSTG). We sincerely thank the authors' works. We also sincerely thank the authors of LargeST (https://github.com/liuxu77/LargeST) for releasing the datasets and baseline evaluation framework.


## Citation
If you find this repository helpful, please consider citing us. \
Meng Wang, Longgang Xiang, Chenhao Wu, Zejiao Wang, Xin Chen, Shaozu Xie, and Ying Luo. 2025. Latent Graph Structure Learning for Large-Scale Traffic Forecasting. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management. (CIKM '25)

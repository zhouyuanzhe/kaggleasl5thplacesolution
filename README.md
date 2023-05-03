# kaggleasl5thplacesolution

This is the source code of the solution of Team ⭐⭐⭐in the prize line⭐⭐⭐ for the competition Google - Isolated Sign Language Recognition (https://www.kaggle.com/competitions/asl-signs/overview).

1. Run makedataset.py to prepare the dataset for dataloader.
2. Run main_zyz.py to run the code ddp.

Our final submission consists of 2 models. One with more parameters with mainly the following settings,
1. embedding size: 480
2. number of head: 16
3. 3 layers of transformer

and another with,
1. embedding size: 240
2. number of head: 16
3. 3 layers of transformer

# MedSam_Pneumonia
This is a project for the Chest X-Ray Image Classification - Pneumonia. The goal of this project is to classify chest X-ray images into two classes: normal and pneumonia.

It leverages MedSAM model. https://github.com/bowang-lab/MedSAM

## Dataset
| Dataset                                                                      | Total images  | Normal vs Pneumonia           |
|------------------------------------------------------------------------------|---------------|-------------------------------|
| [Dataset1](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) | 26684         | Normal: 20672 Pneumonia: 6012 |
| [Dataset2](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | 5856          | Normal: 1341 Pneumonia: 3875  |
| [Dataset3](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images?select=COVID) | 3602         | Normal: 1802 Pneumonia: 1800  |

## Execute the code
Use GPU to run the code for fetching embeddings. We have downloaded all the embeddings of the above 3 datasets and also saved it.

## Reference

```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={arXiv preprint arXiv:2304.12306},
  year={2023}
}
```
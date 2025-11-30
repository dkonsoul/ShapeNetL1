# ShapeNetL1
An L1 Loss adaptation of the original ShapeNet, used for Galaxy Image Reconstruction.

This repository contains the code used in my thesis. It uses the code of three other projects: galaxy2galaxy, ShapeNet and SUNet. Galaxy2galaxy is used to generate a dataset. ShapeNet is the basis for the ShapeNetL1 modification I implemented, while SUNet is adapted to work with my dataset.

This repository is not an official fork of any of these projects.

# Disclaimer and Attributions
This repository uses the code of three other projects. The original upstream of these projects are:
- galaxy2galaxy (cfht2hst_prblm branch)
original project of the branch: https://github.com/fadinammour/galaxy2galaxy.git

- ShapeNet
original project: https://github.com/CosmoStat/ShapeDeconv

- SUNet
original project: https://github.com/FanChiMao/SUNet

ShapeNet is included only as selected files, since only a few of them are needed and has been reorganized. Because this repository does not preserve the original structure, it is not maintained as a fork.

The whole pipeline, requires 3 different environment setups: One for ShapeNet/ShapeNetL1, one for galaxy2galaxy and one for SUNet. Inside the "environments" directory, you will find the corresponding environment.yaml in order to use these projects.

In this repository, you will find the ShapeNetL1 modification, based on the original ShapeNet code, alongside instructions on how to use it.

SUNet modification alongside instructions on how to use it can be found here: {link pending, awaiting license issues}

# Repository Structure
```
/
├── environments #Contains the conda environments for all 3 projects
├── ShapeNetL1 #Contains the original ShapeNet with ShapeNetL1 modifications added
├── scripts #Contains helper scripts to generate the dataset for SUNet and/or generate the same dataset input for all SUNet, ShapeNet and ShapeNetL1 if that is desired, as well as a way to export SUNet's evaluation metrics
├── evaluation #Contains the jupyter notebooks, used to evaluate the results. Files originate from the official ShapeNet project, with modifications where needed
└── HowToUse.md #Contains instructions on how to install and use this project
```

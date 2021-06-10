# ANIMAL-SPOT: An Animal-Independent Deep Learning Framework for Bioacoustic Signal Detection

- [General Description](#general descritpion)
- [Reference](#reference)
- [License](#license)
- [System Requirements](#system-requirements)
- [OS-Dependent Installation](#setting-up-the-development-environment)
- [Data Preparation](#license)
- [Network Training](#license)
- [Network Prediction](#license)
- [Network Evaluation](#license)
- [Example Data Corpus](#license)

# General Description
ANIMAL-SPOT is an animal-independent deep learning software framework addressing various bioacoustic signal identifcation scenarios, such as: (1) binary target/noise detection, (2) multi-class species identification, and (3) multi-class call type recognition. ANIMAL-SPOT is a ResNet18-based Convolutional Neural Network (CNN), taking inspiration from ORCA-SPOT, a ResNet18-based CNN applied to killer whale sound type versus background noise detection (see https://www.nature.com/articles/s41598-019-47335-w). ANIMAL-SPOT was evaluated performing binary target/noise detection on 10 different species and 1 genus scattered around the chordate phylum including: cockatiel (Nymphicus hollandicus), Sulphur-crested cockatoo (Cacatua galerita), Peach-fronted conure
 (Eupsittula aurea), Monk parakeet (Myiopsitta monachus), Golden- and Blue-winged warbler (Vermivora chrysoptera and Vermivora cyanoptera) (genus), Chinstrap penguin
(Pygoscelis antarcticus), Atlantic cod (Gadus morhua), Harbour seal (Phoca vitulina), Killer whale
(Orcinus orca), Pygmy pipistrelle (Pipistrellus pygmaeus), and chimpanzee (Pan troglodytes). Furthermore multi-class species identification was conducted regarding the previous genus (warbler species) in order to further distinguish between the warbler vocalizations. In addition, multi class call type classification was performed for monk parakeets. Due to the large repertoire of various parameterization options provided by ANIMAL-SPOT, the processing of different bioacoustic signal identification scenarios is possible next to the handling of distinct animal species, independent of their respective vocalization characteristics. Such flexibility and versatility leads to broad applicability in the research field of bioacoustics.

# Reference
If ANIMAL-SPOT is used within your own research, please cite the following publication (currently under review at Nature, bibtex will be updated upon acceptance): 

```"Deep Learning Enables Animal-Independent Signal Detection and Classification"```


```
@article{bergler:2021,
author = {Bergler, Christian and Smeele, Simeon Q and Tyndel, Stephen A and Ortiz, Sara T and Kalan, Ammie K. and Cheng, Rachael Xi and Brinkløv, Signe and Osiecka, Anna N and Tougaard, Jakob and Jakobsen, Freja and Wahlberg, Magnus and Nöth, Elmar and Maier, Andreas and Klump, Barbara C},
year = {2021},
month = {},
pages = {},
title = {Deep Learning Enables Animal-Independent Signal Detection and Classification},
volume = {},
journal = {Nature - UNDER REVIEW},
doi = {}
}
```
# License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

# System Requirements
The entire software framework of ANIMAL-SPOT was initially designed, implemented, and verified on Linux (Mint). In general, there exist no special hardware requirements (state-of-the-art desktop computer or laptop), 
but the usage of a graphics processing unit (GPU) is strongly recommended as it significantly speeds up the process of network training and prediction, rather than running
the entire operations on the central processing unit (CPU), however, it is possible. In this study, a mid-range GPU (Nvidia GTX 1080) was used, which led to a real-time factor of 1/25, e.g. predicting/machine-annotating a 45 minute unseen recording within ~2 minutes, compared to > 20-30 minutes on CPU. 
The same applies to network training. Whereas networks, training on small data corpora including just a few thousand samples, using GPU support, are ready in a few hours, CPU-based training might take days depending on the CPU performance.
Considering the large number of training runs to fine-tune and optimize models, while identifying valid parametric constellations, this leads to a considerable time expenditure.

## Operating System
ANIMAL-SPOT is supported on Linux, Windows, and MacOS due to it is a Python-based core framework, which is supported on all operating systems.

## Required Software Packages
ANIMAL-SPOT is a deep learning framework developed in PyTorch, requiring an installation of the following software packages and modules (all versions named below were utilized within this study):

Python (>=3.8), PyTorch Package, including Torch (1.8.1), Torchvision (0.9.1), Torchaudio (0.8.1), 
Librosa (0.8.0), TensorboardX (2.1), Matplotlib (3.3.3), Soundfile (0.10.3.post1), Scikit-Image (0.18.1), Six (1.15.0), Opencv-Python (4.5.1.48), Pillow (8.2.0)

Python: https://www.python.org/downloads/

PyTorch: https://pytorch.org/get-started/locally/

# OS-Dependent Installation
Within the ANIMAL-SPOT framework (see Installation Folder) operating-system-based installation scripts are provided. 
All required software packages are installed automatically within a virtual environment on the respective operating system.
Those installation scripts are of course optional and should provide support especially regarding non-computer science operators. 
An installation of all needed software packages, listed above, can of course also be conducted in a user-specific and individual fashion. 

##Linux
Open linux terminal and execute the installation script "install.sh" (see Installation folder)

```./install *YourInstallationPath* *PyTorchVersion* *TorchVisionVersion*```

e.g. 

```./install /home/myVirtualEnv/ 1.8.1  0.9.1```

##Windows
TODO

##MacOs
TODO

# Data Preparation
ANIMAL-SPOT requires annotated data excerpts (.wav samples) in sufficient quantity (the more the better) for each category/class, 
depending on the bioacoustic signal identification scenario (e.g. target animal vocalizations vs noise and/or different species/call types).
The annotated bioacoustic data samples do not have to be of equal length, but should be labeled with similar temporal context 
(e.g. vocalizations of ~2-10ms for bats compared to several 100ms long bird calls strongly impacts model performance). In order to
properly load and preprocess your data via ANIMAL-SPOT, e.g. automatic class identification/assignment, next to an appropriate data split into a 
network training, validation, and test corpus, followed by an adequate and successful network training,
the filename structure for each .wav excerpt has to follow a general structure, listed here (see also the example corpus, described below):

## General Data and Directory Structure

### Data Structure
Each training sample has to be an annotated and extracted audio file (.wav format) fulfilling subsequent filename structure:

```Filename Template: CLASSNAME-LABELINFO_ID_YEAR_TAPENAME_STARTTIMEMS_ENDTIMEMS.wav```

The entire filename consists of 6 elements, separated via the "_" symbol. Consequently this type of symbol is not allowed within the
strings, cause it is considered as delimiter. The "-" symbol can be used as element to concatenate several strings within one
element, e.g. TAPENAME = REC-Vancouver-Version1.1, which is more clear, rather than RECVancouverVersion1.1 (but both is possible).

1st-Element: CLASSNAME-LABELINFO = The first part CLASSNAME has to be the name of the respective class, 
whereas the second part LABELINFO is optional, and could be used to provide additional label information,
e.g. "orca-podA4callN7", followed by "_ID_YEAR...". If LABELINFO is not used, e.g. "orca-" followed by "_ID_YEAR...",
it is still important to keep the "-" symbol after CLASSNAME, as "-" first occurrence acts as delimiter,
however, only in case of the 1st-Element to ensure a proper class label identification. Conosequently CLASSNAME ends after
the first occurrence of "-".

2nd-Element: ID = unique ID (natural number) to identify the audio clip

3rd-Element: YEAR = year of the tape when it has been recorded

4th-Element: TAPENAME = name of the recorded tape, which is important to do a proper data split into training, 
validation, and test by using the TAPENAME and YEAR as joint unique identifier, in order to avoid that samples of the same 
recording (year and tape) are spread over the distributions. Therefore it is important to include many excerpts from 
different tapes. If your recorded data material does not provide such a sufficient large data/tape variety, the YEAR and 
TAPENAME information can also be artificially renamed to simulate unique identifiers, but data from the same tapes will 
be mixed across the distributions, which may affect network performance later on.

5th-Element: STARTTIMEMS = start time of the audio clip in milliseconds with respect 
to the original recording (natural number)

6th-Element: ENDTIMEMS = end time of the audio clip in milliseconds with respect 
to the original recording (natural number)

Examples of valid filenames:

call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919

CLASSNAME = call, LABELINFO = Orca-A12, ID = 929, YEAR = 2019, TAPENAME = Rec-031-2018-10-19-06-59-59-ASWMUX231648, 
STARTTIMEMS = 2949326, ENDTIMEMS = 2949919

noise-_2381_2010_101BC_149817_150055.wav

CLASSNAME = noise, LABELINFO = , ID = 2381, YEAR = 2010, TAPENAME = 101BC, 
STARTTIMEMS = 149817, ENDTIMEMS = 150055

#### Binary Target/Noise Segmentation
If ANIMAL-SPOT is used for binary target sound versus noise detection, a special feature of the 
filename structure mentioned above must be taken into account. The target class requires the 
CLASSNAME "target", whereas all elements of the noise class should use the CLASSNAME "noise". 
It is important that only these two CLASSNAMES are used consistently to ensure that ANIMAL-SPOT
and its integrated intelligent data preprocessing only identify 2 classes.
 
Examples of valid filenames:

target-monkparakeet_929_2021_TapeA123_2949326_2949919

target-_1000_2001_BaX398_10005000_10003000

noise-underwater_900_2005_CX100_5000_5500

#### Multi-Class Species/Call Type Classification
If ANIMAL-SPOT is used for multi-class species and/or call type classification each class/category has
to be defined by choosing a proper CLASSNAME, whereas every string is possible. ANIMAL-SPOT will automatically identify
each class by its corresponding CLASSNAME.
 
Examples of valid filenames:

alarm-monkparakeet_929_2021_TapeA123_2949326_2949919

N9orca-_1000_2001_BaX398_10005000_10003000

noctule-batspecies_900_2005_CX100_5000_5500

### Directory Structure
ANIMAL-SPOT integrates its own training, validation, and test split procedure in order to properly partionate
the corresponding data archive. 
The entire data (all .wav files) could be either stored under one single main folder or distributed within certain subfolders
under the corresponding main folder, e.g.

ANIMAL-DATA (main folder)

- class1filename1.wav, class2filename2.wav, class3filename1.wav, ... , class1filenameN.wav

ANIMAL-DATA (main folder)

- NOISE (sub-folder)
  - noisefile1.wav, noisefile2.wav, ... , noisefileN.wav
- GOLDENWARBLER (sub-folder)
  - golden1.wav, golden2.wav, ... , goldenN.wav
- BLUEWARBLER (sub-folder)
  - blue1.wav, blue2.wav, ... , blueN.wav
- (...)


In both cases ANIMAL-SPOT will create the datasplit automatically (default: 70% training, 15% validation, 15% testing).
ANIMAL-SPOT creates for each folder three CSV files (train.csv, val.csv, and test.csv), representing each folder-specific data partition. Those directory-specific CSV files are merged and stored together, according to the partition,
in one "train", "val", and "test" file, stored within the main folder. If there exist no sub-folder (see above - option 1),
all files (.csv and train, val, test) will be listed in the main folder. In case of multiple sub-folders, each sub-folder contains 
"train.csv", "val.csv", and "test.csv", whereas the overall "train", "val", and "test" file is located in the main folder.
ANIMAL-SPOT guarantees that no audio files of a single year/tape combination (unique identifier, see Data Structure) 
are spread across training, validation, and testing. Consequently it moves all files of one year/tape pair into only one of the three partitions.
According to the data distribution it might be the case that a certain class might not be present in each partition, which should be avoided.


# Network Training
Once data preparation and processing is completed, the training of the model can be started.
For this purpose, only the user-, data-, and system-specific network hyper-parameters must be defined 
within the corresponding "config" file, located in the "TRAINING" folder. Afterwards the "start_training.py" script 
has to be executed on the corresponding operating system.

A successful network training example (log-file output), utilizing the example data corpus, can be seen [here](OUTPUT/TRAIN.log)!

##Linux
In a first step the previous installed virtual python environment has to be activated (in case a virtual environment is used). 
Depending on the previous user-specific installation this may vary. According to the installation example above the virtual environment can be activated
via open the linux terminal, navigating to the "bin" folder located  under the previous chosen installation path of the environment 
(see above - *YourInstallationPath*), and execute the following terminal command:

```source activate```

Your virtual environment should be activated now only within this terminal sesssion. Therefore the final training
command has also to be launched utilizing this terminal window. 

In a second step open the "config" file within the "TRAINING" folder and adjust all network hyper-parameters
according to your data and training scenario. 

Once the entire config file is setup properly network training
can be started via executing the python script "start_training.py", located within the "TRAINING" folder,
using the previous opened terminal, via:

```./start_training.py```

NOTE: the config file has to be in the same directory as the "start_training.py".

##Windows
TODO

##MacOs
TODO

# Network Prediction
Once model training is finished the corresponding "ANIMAL-SPOT.pk" file (= model) can be applied to unseen data.
Network prediction can be performed on (1) multiple audio files (.wav), stored in a single folder, or (2) a single audio
excerpt (.wav). Depending on the chosen prediction window length and step-size, the model predicts each frame and returns 
the corresponding class and network confidence (probability). Before network prediction can be started all user-, data-, and system-specific network hyper-parameters must be defined 
within the corresponding "config" file, located in the "PREDICTION" folder. Afterwards the "start_prediction.py" script 
has to be executed on the corresponding operating system.

A successful network prediction example (log-file output), utilizing a single audio file of the example data corpus, can be seen here!

##Linux

In a first step the previous installed virtual python environment has to be activated (in case a virtual environment is used). Depending on the previous
user-specific installation this may vary. According to the installation example above the virtual environment can be activated
via open the linux terminal, navigating to the "bin" folder located  under the previous chosen installation path of the environment 
(see above - *YourInstallationPath*), and execute the following terminal command:

```source activate```

Your virtual environment should be activated now only within this terminal sesssion. Therefore the final model prediction
command has also to be executed utilizing this terminal window. 

In a second step open the "config" file within the "PREDICTION" folder and adjust all network hyper-parameters
according to your data and training scenario. 

Once the entire config file is setup properly network training
can be started via executing the python script "start_prediction.py", located within the "PREDICTION" folder,
using the previous opened terminal, via:

```./start_prediction.py```

NOTE: the config file has to be in the same directory as the "start_training.py".

##Windows
TODO

##MacOs
TODO

# Network Evaluation
Once model prediction is finished the corresponding "*.predict_output.log" files can be used as input to generate
an annotation file according to the format readable and interpretable via RAVEN (https://ravensoundsoftware.com/). Before network evaluation can be started all user-, data-, and system-specific network hyper-parameters must be defined 
within the corresponding "config" file, located in the "EVALUATION" folder. Afterwards the "./start_evaluation.py" script 
has to be executed on the corresponding operating system.

A successful network evaluation example (annotation file in RAVEN format), utilizing a single prediction file as input, can be seen here!

In addition to that, ANIMAL-SPOT is monitoring, documenting and storing the entire training, validation, as well as the final testing procedure.
Therefor, ANIMAL-SPOT reports various machine learning metrics, such as accuracy, F1-score, recall, false-positive-rate, precision, loss, learning rate, etc., 
next to network input spectrogram examples and their respective classification hypotheses. All information and documented results
can be reviewed via tensorboard, together with the automatically generated "summaries" folder (see directory path for summary folder within
the config file for network training). The following command has to be executed:

```tensorboard --logdir /directory_to_model/summaries/```

##Linux
In a first step open the "config" file within the "EVALUATION" folder and adjust all network hyper-parameters
according to your scenario.  

Once the entire config file is setup properly network evaluation can be conducted via executing the python script 
"start_evaluation.py", located within the "EVALUATION" folder, using the terminal, via:

```./start_evaluation.py```

NOTE: the config file has to be in the same directory as the "start_evaluation.py".

##Windows
TODO

##MacOs
TODO

# Example Data Corpus
TODO
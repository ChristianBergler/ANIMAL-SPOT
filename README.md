# ANIMAL-SPOT enables animal-independent signal detection and classification using deep learning

- [General Description](#general-description)
- [Reference](#reference)
- [License](#license)
- [System Requirements](#system-requirements)
- [OS-Dependent Installation - TBD](#os-dependent-installation)
- [Example Data Corpus and Recording - TBD](#example-data-corpus-and-recording)
- [Data Preparation](#data-preparation)
- [Network Training](#network-training)
- [Network Prediction](#network-prediction)
- [Network Evaluation](#network-evaluation)
- [FAQ](#FAQ)

# General Description
ANIMAL-SPOT is an animal-independent deep learning software framework that addresses various bioacoustic signal identifcation scenarios, such as: (1) binary target/noise detection, (2) multi-class species identification, and (3) multi-class call type recognition. ANIMAL-SPOT is a ResNet18-based Convolutional Neural Network (CNN), taking inspiration from ORCA-SPOT, a ResNet18-based CNN applied to killer whale sound type versus background noise detection (see https://www.nature.com/articles/s41598-019-47335-w). ANIMAL-SPOT's performance
was evaluated detecting binary target/noise signals using 10 different species and 1 genus scattered around the chordate phylum: cockatiel **(Nymphicus hollandicus)**, Sulphur-crested cockatoo **(Cacatua galerita)**, Peach-fronted conure
**(Eupsittula aurea)**, Monk parakeet **(Myiopsitta monachus)**, Blue- and Golden-winged warbler **(Vermivora cyanoptera and Vermivora chrysoptera)** (genus), Chinstrap penguin
**(Pygoscelis antarcticus)**, Atlantic cod **(Gadus morhua)**, Harbour seal **(Phoca vitulina)**, Killer whale
**(Orcinus orca)**, Pygmy pipistrelle **(Pipistrellus pygmaeus)**, and chimpanzee **(Pan troglodytes)**. 
Additionally multi-class species identification was conducted on the dataset including both warbler species in order to further distinguish
between the two species. In addition, multi class call type classification was performed for monk parakeets. 
ANIMAL-SPOT provides a large repertoire of parameterization options, enabling the 
the processing of different bioacoustic signal identification scenarios, independent
of the characteristics of the target vocalizations. Such flexibility and versatility leads to broad applicability 
in the field of bioacoustics research.

# Reference
If ANIMAL-SPOT is used within your own research, please cite the following publication, available here https://www.nature.com/articles/s41598-022-26429-y :

```"ANIMAL-SPOT enables animal-independent signal detection and classification using deep learning"```


```
@article{BerglerAnimalSpot:2022,
author = {Bergler, Christian and Smeele, Simeon and Tyndel, Stephen and Barnhill, Alexander and Torres Ortiz, Sara and Kalan, Ammie and Cheng, Rachael Xi and Brinkløv, Signe and Osiecka, Anna and Tougaard, Jakob and Jakobsen, Freja and Wahlberg, Magnus and Noeth, Elmar and Maier, Andreas and Klump, Barbara},
year = {2022},
month = {12},
pages = {21966},
title = {ANIMAL-SPOT enables animal-independent signal detection and classification using deep learning},
volume = {12},
journal = {Scientific Reports},
doi = {10.1038/s41598-022-26429-y}
}
```
# License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

# System Requirements
The software framework of ANIMAL-SPOT was initially designed, implemented, and verified on Linux (Mint), but can be
used on Windows and MacOS as well. In general, no special hardware requirements exist 
(state-of-the-art desktop computer or laptop is sufficient), but the usage of a graphics processing unit 
(GPU) is strongly recommended as it significantly speeds up the process of network training 
and prediction. Running the entire operations on the central processing unit (CPU),
is, however, possible. In this study, a mid-range GPU (Nvidia GTX 1080) was used, which 
processed one unseen recording (duration of track was ~45 minutes) within ~2 minutes, compared 
to > 20-30 minutes on CPUs. 
The same applied to network training. Networks trained on small data corpora (including 
only a few thousand samples) are ready in a few hours whe using GPU support, while CPU-based
training might take days depending on the CPU performance. As a large number of training runs
is necessary to identify valid parameter constellations and fine-tune and optimize models, the use of a GPU
leads to considerable time saving.

## Operating System
ANIMAL-SPOT is available on Linux, Windows, and MacOS due to its Python-based core framework, which is supported on all operating systems.

## Required Software Packages
ANIMAL-SPOT is a deep learning framework developed in PyTorch, requiring the
installation of the following software packages and modules 
(during development of ANIMAL-SPOT, the following software packages/versions were used and are compatible):

Python (>=3.8), PyTorch Package, including Torch (1.11.0+cu113), Torchvision (0.12.0+cu113), Torchaudio (0.11.0+cu113), 
Librosa (0.8.0), TensorboardX (2.1), Matplotlib (3.3.3), Soundfile (0.10.3.post1), Scikit-Image (0.18.1), Six (1.15.0), Opencv-Python (4.5.1.48), Pillow (8.2.0)

Python: https://www.python.org/downloads/

PyTorch: https://pytorch.org/get-started/locally/

# OS-Dependent Installation
Apart from the source code as well as an example data corpus, the ANIMAL-SPOT framework also provides
an automatic operating-system-independent installation procedure to setup and install all software requirements
in order to properly deploy the software. Available soon!

# Example Data Corpus and Recording
Next to all the guidelines/scripts and source code, the proposed and final example data corpus for target vs. noise detection on Monk parakeets (Myiopsitta monachus).
This dataset is exactly the one which was used in the manuscript for Monk parakeet target/noise detection, including an overall
number of 6,745 audio files, whereas 3,133 belong to the target class and 3,612 to the noise category (see Table 1 in the Manuscript). In addition
exactly the same data split is provided. This data archive can be used right away for training and evaluating the network, in order to verify a valid installation, data preparation, and appropriate model training.
In addition to this data archive (cropped audio files), an unseen and raw audio recording on Monk parakeets is provided in order to also test
a valid network prediction procedure. For specific questions about the dataset or any further interest, please refer
to Simeon Q. Smeele (co-author). When using this data in any form, the following citations are mandatory:

```
@article{BerglerAnimalSpot:2022,
author = {Bergler, Christian and Smeele, Simeon and Tyndel, Stephen and Barnhill, Alexander and Torres Ortiz, Sara and Kalan, Ammie and Cheng, Rachael Xi and Brinkløv, Signe and Osiecka, Anna and Tougaard, Jakob and Jakobsen, Freja and Wahlberg, Magnus and Noeth, Elmar and Maier, Andreas and Klump, Barbara},
year = {2022},
month = {12},
pages = {21966},
title = {ANIMAL-SPOT enables animal-independent signal detection and classification using deep learning},
volume = {12},
journal = {Scientific Reports},
doi = {10.1038/s41598-022-26429-y}
}
```

```
@article {SmeeleMonkParakeet:2022,
author = {Smeele, Simeon Q. and Tyndel, Stephen A. and Aplin, Lucy M. and McElreath, Mary Brooke},
title = {Multi-level analysis of monk parakeet vocalisations shows emergent dialects between cities in the European invasive range},
elocation-id = {2022.10.12.511863},
year = {2022},
doi = {10.1101/2022.10.12.511863},
publisher = {Cold Spring Harbor Laboratory},
URL = {https://www.biorxiv.org/content/early/2022/10/16/2022.10.12.511863},
journal = {bioRxiv}
}
```

The entire data material (cropped audio files, raw unseen prediction tape, and data split) can be downloaded here: TBD (soon!)

# Data Preparation
ANIMAL-SPOT requires annotated data excerpts (.wav samples) in sufficient quantity for each category/class, depending on the bioacoustic signal identification scenario 
(e.g. target animal vocalizations vs. noise, different specie/call type identification). In this study an average amount of ~5,000 human-annotated training data samples per species was 
reached, while ignoring the largest and smallest dataset, and calculating the mean across the remaining animal-specific data archives, but of course, the more representative 
training material, the better network generalization. The annotated bioacoustic data samples do not have to be of equal length (variable durations are possible). However, 
during training a fixed sequence length has to be chosen, which should be close to the average duration of the involved animal-specific target vocalization(s). 
In general, it is important that the human-labeled animal vocalizations are precise, whereas small signal contents before and after the actual 
vocalization are acceptable. The same applies to special noise events. In order to properly load and preprocess your data via ANIMAL-SPOT, the filename structure for each .wav excerpt has to follow the same
structure, listed here (see also the example corpus, described below).

## General Data and Directory Structure
The following section describes the required data and directory structure. Furthermore, the provided example data corpus (see below, *Example Data Corpus*)
serves an additional evidence for a valid and correct data preparation. 

### Data Structure
Each training sample has to be an annotated and extracted audio file (.wav format) with the below filename structure:

```Filename Template: CLASSNAME-LABELINFO_ID_YEAR_TAPENAME_STARTTIMEMS_ENDTIMEMS.wav```

The entire filename consists of 6 elements, separated via the "\_" symbol. Consequently this type of symbol is not allowed within the
strings, because it is considered as delimiter. Moreover, the "-" symbol has an important and special meaning. 
For separation of the two filename parts, *CLASSNAME* and *LABELINFO* (see filename template), the "-" symbol is mandatory 
and acts as delimiter in order to identify the *CLASSNAME*. Within all other parts of the filename template (e.g. *ID*, *YEAR*, *TAPENAME*) the
"-" symbol can be used as element to concatenate several strings, as any other symbol, except "\_".

**1st-Element: CLASSNAME-LABELINFO** = The first part *CLASSNAME* has to be the name of the respective class, 
whereas the second part *LABELINFO* is optional, and could be used to provide additional label information,
e.g. "orca-podA4callN7", followed by "_ID_YEAR...". If LABELINFO is not used, it is still important to keep the "-" symbol after *CLASSNAME*, 
as the first occurrence of "-" acts as delimiter. *CLASSNAME* ends after
the first occurrence of "-".

**2nd-Element: ID** = unique *ID* (natural number) to identify the audio clip

**3rd-Element: YEAR** = the year the tape has been recorded

**4th-Element: TAPENAME** = name of the recorded tape. This is important for a proper split of the data into training, validation, and test.
This is achieved by using the *TAPENAME* and *YEAR* as joint unique identifier, in order to avoid that samples of the same 
recording (year and tape) are spread over the distributions. It is therefore important to include many excerpts from 
different tapes, in order to ensure a proper and automatic data split. 

**5th-Element: STARTTIMEMS** = start time of the audio clip in milliseconds within the original recording (natural number)

**6th-Element: ENDTIMEMS** = end time of the audio clip in milliseconds within the original recording (natural number)

**Examples of valid filenames:**

*call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919*

CLASSNAME = call, LABELINFO = Orca-A12, ID = 929, YEAR = 2019, TAPENAME = Rec-031-2018-10-19-06-59-59-ASWMUX231648, 
STARTTIMEMS = 2949326, ENDTIMEMS = 2949919

*noise-_2381_2010_101BC_149817_150055.wav*

CLASSNAME = noise, LABELINFO = , ID = 2381, YEAR = 2010, TAPENAME = 101BC, 
STARTTIMEMS = 149817, ENDTIMEMS = 150055

#### Binary Target/Noise Segmentation
If ANIMAL-SPOT is used for binary target sound vs, noise detection, a special feature of the 
filename structure mentioned above must be used. The target class requires the 
*CLASSNAME* *target*, and all elements of the noise class the *CLASSNAME* *noise*. 
It is important that only these two *CLASSNAMES* are used to ensure that ANIMAL-SPOT
and its integrated intelligent data preprocessing only identify two classes.
 
**Examples of valid filenames:**

*target-monkparakeet_929_2021_TapeA123_2949326_2949919*

CLASSNAME = target, LABELINFO = monkparakeet, ID = 929, YEAR = 2021, TAPENAME = TapeA123, 
STARTTIMEMS = 2949326, ENDTIMEMS = 2949919

*target-_1000_2001_BaX398_10005000_10003000*

CLASSNAME = target, LABELINFO = , ID = 1000, YEAR = 2001, TAPENAME = BaX398, 
STARTTIMEMS = 10005000, ENDTIMEMS = 10003000

*noise-underwater_900_2005_CX100_5000_5500*

CLASSNAME = noise, LABELINFO = underwater, ID = 900, YEAR = 2005, TAPENAME = CX100, 
STARTTIMEMS = 5000, ENDTIMEMS = 5500

#### Multi-Class Species/Call Type Classification
If ANIMAL-SPOT is used for multi-class species and/or call type classification each class/category has
to be defined by choosing a CLASSNAME (every string is possible) that is used consistently. ANIMAL-SPOT will automatically identify
each class by its corresponding CLASSNAME.
 
**Examples of valid filenames:**

*alarm-monkparakeet_111_2021_B123_1249326_1349919*

CLASSNAME = alarm, LABELINFO = monkparakeet, ID = 111, YEAR = 2021, TAPENAME = B123, 
STARTTIMEMS = 1249326, ENDTIMEMS = 1349919

*N9orca-_1000_2001_BaX398_10005000_10003000*

CLASSNAME = N9orca, LABELINFO = , ID = 1000, YEAR = 2001, TAPENAME = BaX398, 
STARTTIMEMS = 10005000, ENDTIMEMS = 10003000

*noctule-batspecies_910_2005_CX100_5000_7500*

CLASSNAME = noctule, LABELINFO = batspecies, ID = 2005, YEAR = 2021, TAPENAME = CX100, 
STARTTIMEMS = 5000, ENDTIMEMS = 7500

### Directory Structure
ANIMAL-SPOT integrates its own training, validation, and test split procedure in order to properly partition the corresponding data archive.
The entire dataset (all .wav files) could be either stored under one single main folder or distributed within certain subfolders
under the corresponding main folder, e.g.

**ANIMAL-DATA (main folder)**

- *golden1.wav, golden2.wav, blue1.wav, blue2.wav, noisefile1.wav, noisefile2.wav, ... , goldenN.wav, blueN.wav, noisefileN.wav*

**ANIMAL-DATA (main folder)**

- **NOISE (sub-folder)**
  - *noisefile1.wav, noisefile2.wav, ... , noisefileN.wav*
- **GOLDENWARBLER (sub-folder)**
  - *golden1.wav, golden2.wav, ... , goldenN.wav*
- **BLUEWARBLER (sub-folder)**
  - *blue1.wav, blue2.wav, ... , blueN.wav*
- **(...)**


In both cases ANIMAL-SPOT will create the datasplit automatically (default: 70% training, 15% validation, 15% testing).
ANIMAL-SPOT creates three CSV files (train.csv, val.csv, and test.csv) for each folder, in case the above mentioned data separation 
rule - samples of a single recording tape will not be distributed across various partitions - could
be applied successfully. Otherwise data of a specific sub-folder is not present within all partitions and therefore less than three
.csv files are generated for that sub-folder. Those directory-specific CSV files are merged and stored together, according to the partition,
in one *train, val, and test* file, stored within the main folder. If no sub-folder exist (see above - option 1),
all files (.csv and train, val, test) will be listed in the main folder. In case of multiple sub-folders, each sub-folder contains 
*train.csv, val.csv, and test.csv*, whereas the overall *train, val, and test* file is located in the main folder.
ANIMAL-SPOT guarantees that no audio files of a single year/tape combination (unique identifier, see Data Structure) 
are spread across training, validation, and testing. Consequently it moves all files of one year/tape pair into only one of the three partitions.
According to the data distribution it might be the case that a certain class might not be present in each partition, which should be avoided.
Either by deleting the train, val, test files, together with all .csv files, and start a new training, which will automatically 
initiate a new random data split including all the required files, or crosscheck the year/tape (unique identifier) combinations for that specific class. Dependent on the above folder structure
a successful data split would be,

**ANIMAL-DATA (main folder)**

- *golden1.wav, golden2.wav, blue1.wav, blue2.wav, noisefile1.wav, noisefile2.wav, ... , goldenN.wav, blueN.wav, noisefileN.wav, **train.csv, val.csv, test.csv, train, val, test***

**ANIMAL-DATA (main folder)**

**train, val, test**
- **NOISE (sub-folder)**
  - *noisefile1.wav, noisefile2.wav, ... , noisefileN.wav, **train.csv, val.csv, test.csv***
- **GOLDENWARBLER (sub-folder)**
  - *golden1.wav, golden2.wav, ... , goldenN.wav, **train.csv, val.csv, test.csv***
- **BLUEWARBLER (sub-folder)**
  - *blue1.wav, blue2.wav, ... , blueN.wav, **train.csv, val.csv, test.csv***
- **(...)**

# Network Training
Once data preparation and processing is completed, the training of the model can be started.
For this purpose, the user-, data-, and system-specific network hyper-parameters must be defined 
within the corresponding *config* file, located in the *TRAINING* folder. Each config parameter is explained in detail
within the config file. Furthermore, the config file specifies empirical values for certain network hyper-parameters, 
acting as a general guideline. Afterwards the *start_training.py* script  has to be executed on the 
corresponding operating system.

A successful network training example (log-file output), utilizing the example data corpus, will be available soon!

##Linux, MacOS, and Windows
Open the *config* file within the *TRAINING* folder and adjust all network hyper-parameters
according to your data and training scenario, while considering parameter explanations, use-case-specific settings, as 
well as default values, listed within the *config* file (see also Supplementary Table 2 within the manuscript for species-specific
reference values). Once the entire config file is set up, the network training can be started via executing 
the python script *start_training.py*, located within the *TRAINING* folder, utilizing the terminal, via:

```python start_training.py *PathToConfigFile*```

**NOTE:** Windows user: set the *num_workers* parameter within the *config* file to 0, due to multiprocessing restrictions in Windows.

# Network Prediction
Once model training is finished the resulting *ANIMAL-SPOT.pk* file (= model) can be applied to unseen data.
Network prediction can be performed on (1) multiple audio files (.wav), stored in a single folder, or (2) a single audio
excerpt (.wav). All required user-, data-, and system-specific network prediction parameters are explained and listed within the prediction config file (see *PREDICTION* folder), 
similar to the training procedure (see section *Network Training*). Depending on the chosen prediction window length and step-size, the model predicts each frame and returns 
the respective class and network confidence (probability). Afterwards the *start_prediction.py* script has to be executed on the corresponding operating system.

A successful network prediction example (log-file output) for binary detection, utilizing a single audio file of the example data corpus, will be available soon!

##Linux, MacOS, and Windows
Open the *config* file within the *PREDICTION* folder and adjust all network prediction hyper-parameters
according to your data and prediction scenario, while considering parameter explanations, use-case-specific settings, as 
well as default values, listed within the *config* file. 

Once the entire config file is set up network prediction can be started via executing 
the python script *start_prediction.py*, located within the *PREDICTION* folder,
utilizing the terminal, via:

```python start_prediction.py *PathToConfigFile*```

**NOTE:** Windows user: set the *num_workers* parameter within the *config* file to 0, due to multiprocessing restrictions in Windows.

# Network Evaluation
Once model prediction is finished, the resulting **.predict_output.log* files can be used as input to generate
an annotation file, which can be opened via RAVEN (https://ravensoundsoftware.com/). All required user-, data-, 
and system-specific network evaluation parameters are explained and listed within the evaluation config file 
(see *EVALUATION* folder), similar to the training procedure (see section *Network Training*).
Afterwards the *start_evaluation.py* script has to be executed on the corresponding operating system.

A successful network evaluation example (annotation file in RAVEN format), utilizing a single prediction file as input, will be available soon!

ANIMAL-SPOT is monitoring, documenting and storing the entire training, validation, as well as the final testing 
procedure. It reports various machine learning metrics, such as accuracy, F1-score, recall, 
false-positive-rate, precision, loss, learning rate, etc., next to network input spectrogram examples and 
their respective classification hypotheses. All information and documented results can be reviewed via 
tensorboard (toolkit which provides a visualization framework to monitor and visualize the entire network training, 
validation, and testing procedure), together with the automatically generated *summaries* folder 
(see network training config file, path to summary folder). The following command has to be executed:

```tensorboard --logdir /*PathToNetworkSummaryFolder*/summaries/```

##Linux, MacOS, and Windows
In a first step open the *config* file within the *EVALUATION* folder and adjust all network hyper-parameters
according to your scenario.  

Once the entire config file is set up, the network evaluation can be conducted via executing the python script 
*start_evaluation.py*, located within the *EVALUATION* folder, using the terminal, via:

```python start_evaluation.py *PathToConfigFile*```


# FAQ
Common user-, use-case-, system-, as well as data-specific questions and problems are clarified and answered here!

(1) After PyTorch installation in Windows there might be an error when trying to *import torch* within the
activated virtual environment. 

Error: 

OSError: [WinError 126] The given module can not be found. Error loading "*YourVirtualEnvInstallationPath*\lib\site-packages\torch\lib\c10.dll" or one of its dependencies.

Solution:

This error can be solved by installing the most recent Microsoft Visual C++ Redistributable version. Choose your
system architecture and download the win exe. After intsallation the error should be gone

Link (last visited, 18.06.2021): https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0
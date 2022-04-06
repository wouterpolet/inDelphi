# inDelphi README

Within this file, we explain how to run our inDelphi adaptation.

# Table of Contents
1. [Introduction](#introduction)
2. [Data Availability](#data)
3. [Directory Structure](#files)
4. [Program Parameters](#params)
5. [Program Execution](#exec)
    1. [Generic Case](#generic)
    2. [Different Execution Cases](#different)
    3. [Pre-Trained Execution Cases](#trained)
    4. [Training Execution Cases](#training)
    5. [Research Question Execution](#RQs)

----
<div id='introduction'/>

## Introduction 


This is a copied repo of https://github.com/maxwshen/indelphi-dataprocessinganalysis with some adaptations for the Machine Learning in Bioinformatics course at TU Delft (CS4260).
Below, please find the appropriate commands to train, predict, or plot the graphs supplied in the report. 

----

<div id='data'/>

## Data Availability

Unfortunately, since some of the data & files used/generated exceeded 100MB, we could not upload these to the git repository.

For replication and completeness purposes, a Google Drive folder is provided.

https://drive.google.com/drive/folders/1GNwWVZT6-ESHYKTV5-Eott3x-xlbipTc

This folder contains the files such as the
- intron: All the human intron sequences  (This must be stored in the following directory: /in/exon_intron/)
- exon: All the human exon sequences  (This must be stored in the following directory: /in/exon_intron/)
- freq_distribution.pkl: The predicted frequency distributions for figure 3f (This must be stored in the following directory: /out/``model_name``/predictions/freq_distribution.pkl)

It is important to note that these can be generated using the application but usually take from 2-to 12 hours, depending on size and complexity.

----

<div id="files"/>

## Directory Structure

Below please find all the folders encapsulated within the submitted package.

- ``author_code``: The original source code used by the inDelphi model with minor amedments (code commenting) for reference purposes
- ``Flowcharts``: All core flowcharts describing this project (also supplied in the Appendix of the report) 
- ``functionality``: All subprocesses and important functionality used for the group & individual research
- ``in``: All input data (including exon/intron, cut sites generated, lib-A sequences and more)
- ``out``: Files & Folders generated during execution of the inDelphi file (trained models)
- ``g_genome_predict.txt``: Pseudo code and commenting describing the authors g_genome_predict file
- ``inDelphi.py``: Main file for our inDelphi execution
- ``Pipfile``: Pip file describing the required packages
- ``Pipfile.lock``: Auto-generated during pip installation 
- ``README.md``: This readme describing important aspects of the program
- ``SupplementaryData.xlsx``: Supplementary Data obtained from the authors paper

---

<div id="params"/>

## Program Parameters

To aid with the program's execution, our main file allows the user to specify some 
functionality they would like to test/change between program runs. Below please find 
a list of all the implemented parameters that can be set and allowable values for said parameters.

| Name         | Allowable Values | Description                                                                                                                                                                                |
|--------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| process      | [3f, 4b, both]   | This is which training/predictions should be carried out (in case of both these are executed in sequence)                                                                                  |
| model_folder | String           | Parent folder name (Within the out directory) containing the trained models. If left empty, the model will be re-trained and new folder will be created                                    |
| load_pred    | True/False       | Load predicted values from the latest execution, if false new predictions will be calculated 																							   |
| new_targets  | True/False       | Boolean variable indicating if a new set of targets should be calculated (only applicable for fig 3f). This process takes some times, since it reads from a 17GB file and reloads 1,003,524|

All boolean variables are defaulted to false in case they are not provided during execution.
In case the model_folder is not provided, the program starts training (a) new model(s).

Some examples of how to use the parameters mentioned below:

- ```python inDelphi.py --process 3f --new_targets True --model_folder 3f_genes```
- ```python inDelphi.py --process 3f --load_pred True --model_folder 3f_genes```
- ```python inDelphi.py --process 4b --model_folder 4b_all_samples```
- ```python inDelphi.py --process both```

----

<div id="exec"/> 

## Program Execution

NB: Please ensure the appropriate files and data have been loaded to avoid unexpected errors.

For more information, please refer to 

<div id="generic"/> 

### Generic Case

1. Install all of the packages and libraries needed using: 
``pipenv install``
   1. Please ensure that the terminal is located in the directory which stores the **inDelphi.py**
2. Identify which parameters to set (as described in the Program Parameters Section)
3. Run the program using : ```python inDelphi.py [some params]``` (and any other parameters desired)

<div id="different"/>

### Different Execution Cases

Below is a brief explanation and examples of the possible different execution cases


----

<div id="trained"/> 

### Pre-Trained Execution Cases

Load the pre-trained model (from the folder called '3f_genes', located under 'out') 

1. ```python inDelphi.py --process 3f --model_folder 3f_genes --new_targets True ```
2. ```python inDelphi.py --process 3f --model_folder 3f_genes --load_pred True ```

In the case of (1), predictions are re-calculated, but prior to re-calculating the prediction, the algorithm loads a new set of random targets (1,003,524) and afterward predicts the distribution of those targets.

In the case of (2), no recalculation is done. It uses the cached files and data and plots the plot.

Load the pre-trained model (from the folder called '4b_all_samples', located under 'out') 

1. ```python inDelphi.py --process 4b --model_folder 4b_all_samples```
2. ```python inDelphi.py --process 4b --model_folder 4b_all_samples --load_pred True ```

In the case of (1), predictions are re-calculated using the cached test datasets (created on model training).

In the case of (2), no recalculation is done. It uses the cached files and data and plots the plot.

Load the pre-trained model (from the folder called 'all_models', located under 'out') 

1. ```python inDelphi.py --process both --model_folder all_models```
2. ```python inDelphi.py --process both --model_folder all_models --load_pred True```

In the case of (1), predictions are re-calculated using the cached test datasets (created on model training).

In the case of (2), no recalculation is done. It uses the cached files and data and plots all the plots.

----

<div id="training"/>

### Training Execution Cases

To train using the code provided, please use one of the following commands:

1. ```python inDelphi.py --process 3f``` -> Creates and trains a new model and predicts frequency distribution for 3f (using the 1M samples already cached). Expected Execution Time: 8 hours.
2. ```python inDelphi.py --process 3f --new_targets=True``` -> Creates and trains a new model and predicts frequency distribution for 3f (using a new 1M samples & caches them). Expected Execution Time: 10/12 hours.
3. ```python inDelphi.py --process 4b``` -> Creates and trains a new model and predicts indel length for 4b (using the subset of mESC and U2OS). Expected Execution Time: 30 min.
4. ```python inDelphi.py --process both``` -> Creates and trains two new models and predicts frequency distribution and indel length for 3f and 4b. Expected Execution Time: 12 hours.
5. ```python inDelphi.py --process both --new_targets=True``` -> Creates and trains two new models and predicts frequency distribution and indel length for 3f and 4b. Expected Execution Time: 12 hours.


---

<div id="RQs"/>

### Research Question Execution

Below please find the commands to execute our research questions and obtain the resutls described in the report.

**Franz:**

**Jonathan:**

Prerequisits: A pre-trained model.

For the purposes of time: ``jon_all_models`` has been provided, where all 3 networks have been pre-trained.

To re-train max & single-task learning frameworks, please supply a trained model from the generic application (as per the details above) and run the command

Command: ``python C:\Development\CS4260\inDelphi\functionality\RQs\Jon\rq.py --model_folder=jon_all_models``

**Wounter:**



---

The model was developed and testing on the following machine details:

Case 1:

- <u>Processor</u>: Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz   2.21 GHz
- <u>RAM</u>: 32.0 GB
- <u>System Type</u>: 64-bit operating system, x64-based processor\
- <u>Operating System</u>: Windows 11 Pro



# Predictive Dead Reckoning

## Installation
This repository requires the following libraries:
```
pytorch
seaborn
matplotlib
numpy
tqdm
scikit-learn
```
## Predictive NN Training and Test Setup
There are a number of scripts here to make setting up for training a slightly easier process.  The training data for position prediction is located in
```
./data/training-data.7z
```
Please note that the files are stored as .7z compressed files using the git large file system.  Please see the notes about installation [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).  The split used for training and testing is also available and can be found in
```
./data/split-training-data.7z
```

You can use this split to replicate our results. If you desire some other arrangement, split the files in *training-data.7z* into two folders.  Currently, the conversion script is expecting
```
./split-test-data
./split-train-data
```
Once that is done, edit the python script `converter.py` and modify the *input* and *output* variables at the top of the file to reflect the desired location of the input files and the formatted output data.

Make sure those directories exist.

 ### Convert the data

There are two convenience scripts that translate the raw position data from ascii text to the format necessary for the NN to process.  If training the single input model, use
 ```
 mk_single_grps.sh
 ```
 Conversely, for the LSTM data format, use
 ```
 mk_grps.sh
 ```

## Training
Training is completed using the script
```
train_prediction_model.py
```

## Testing

 Once the data is prepared, testing is completed using the python script
 ```
 test_models
 ```
 which will test all three model types against the test data -- you may need to confirm the location of the test data.  Results for each run should be written to './test_results' and a summary to 'rmse-results.csv'.

## Graphing

The RMSE error for a single run can be graphed using
```
graph_error.py
```
A comparative RSME error for all runs is created using
```
python graph_rmse_range.py -f rmse-results.csv [-p <prefix>]
```

#Enter the location of your data file in the variable below. The datafile should be a folder that contains pickle files for source and target.

datafile="dataset_initialization/initialization_data "

mkdir iteration0
mkdir iteration0/logs
cp -r $datafile iteration0/data

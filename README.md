# inDelphi reproduction

This is a copied repo of https://github.com/maxwshen/indelphi-dataprocessinganalysis with some adaptations for the Machine Learning in Bioinformatics course at TU Delft.

# Data Availability:
https://drive.google.com/drive/folders/1GNwWVZT6-ESHYKTV5-Eott3x-xlbipTc

Prediction Outcomes Files: extended_prediction_output_20220314_2255.pkl

# Execution Steps and procedures
TODO fill with args and options for exec (should be plug and play for staff)#
Load only pre-trained model 3f named 3f_all_samples in the output folder
and load pre-calculated predictions from the file named:freq_distribution.pkl
(extension is important)

--process=3f --model_folder=3f_all_samples --pred_file=freq_distribution.pkl

Load only pre-trained model 4b named 4b_all_samples in the output folder
and load pre-calculated predictions from the file names: in_del_distribution_mesc.pkl,in_del_distribution_u2os.pkl
(extension is important, should be comma seperated)

--process=4b --model_folder=4b_all_samples --pred_file=in_del_distribution_mesc.pkl,in_del_distribution_u2os.pkl

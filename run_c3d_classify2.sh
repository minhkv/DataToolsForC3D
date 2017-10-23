STARTTIME=$(date +%s)
now=$(date)
echo "$now start classification" >> time_classify_ucf_finetune_split_3.txt
echo "$now start classification" >> log_classify_ucf_finetune_split_3.txt
echo "$now start classification" >> err_classify_ucf_finetune_split_3.txt
python run_c3d_classify2.py >>log_classify_ucf_finetune_split_3.txt 2>> err_classify_ucf_finetune_split_3.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop classification" >> time_classify_ucf_finetune_split_3.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to classify" >> time_classify_ucf_finetune_split_3.txt
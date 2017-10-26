STARTTIME=$(date +%s)
now=$(date)
name="classifier_ucf_finetune_2_fc6"
mkdir -p report_$name
echo "$now start classification" >> time_classify_$name.txt
echo "$now start classification" >> log_classify_$name.txt
echo "$now start classification" >> err_classify_$name.txt
python run_c3d_classify.py >>log_classify_$name.txt 2>> err_classify_$name.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop classification" >> time_classify_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to classify" >> time_classify_$name.txt
mv *$name.* report_$name
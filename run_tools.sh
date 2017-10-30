STARTTIME=$(date +%s)
now=$(date)
# name="feature_extraction_lost_sport1m"
# name="convert_lost_ucf_2_fc6"
# name="test_net_finetune_ucf_3"
name="classify_ucf_1_prob_test"

# run_file=run_c3d_feature_extraction.py
# run_file=run_c3d_convert_bin_to_csv.py
# run_file=run_c3d_test_net.py
# run_file=run_c3d_classify.py
run_file=run_classifier_prob.py

report_folder=report

mkdir -p $report_folder/report_$name
echo "$now start $name" >> $report_folder/time_$name.txt
echo "$now start $name" >> $report_folder/log_$name.txt
echo "$now start $name" >> $report_folder/err_$name.txt
python $run_file $name >>$report_folder/log_$name.txt 2>> $report_folder/err_$name.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop $name" >> $report_folder/time_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to $name" >> $report_folder/time_$name.txt
mv *$name.* $report_folder/*$name.* $report_folder/report_$name

STARTTIME=$(date +%s)
now=$(date)
# name="feature_extraction_longterm_ucf2"
# name="convert_lost_ucf_2_fc6"
# name="test_net_finetune_ucf_3"
# name="test_net_train_error_ucf_2"
# name="classify_sport1m_fc6_rankpool_linear"
name="test_optical_flow_image"

# run_file=run_c3d_feature_extraction.py
# run_file=run_c3d_convert_bin_to_csv.py
# run_file=run_c3d_test_net.py
# run_file=run_c3d_test_train_accuracy.py
# run_file=run_c3d_classify_rankpool.py
# run_file=run_classifier_prob.py
# run_file=run_classify_mid_feature.py
# run_file=AnalyseData.py
run_file=run_create_of_image.py

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

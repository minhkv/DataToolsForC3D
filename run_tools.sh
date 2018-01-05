STARTTIME=$(date +%s)
now=$(date)
# name="feature_extraction_longterm_ucf2"
# name="convert_lost_ucf_2_fc6"
name="create_flow_hmdb51_finetune_1"
# name="test_net_train_error_ucf_2"
# name="classify_flow_60000_s1_prob_verify"
# name="feature_extract_ucf"
# name="test_mica_3600_v5"
# name="testnet_mica_dataset_fix_label"
# name="feature_mica_dataset_fix_label_dense"
# name="classify_mica_fix_label_video"

# run_file=run_c3d_feature_extraction.py
# run_file=run_c3d_convert_bin_to_csv.py
# run_file=run_c3d_test_net.py
# run_file=run_c3d_test_train_accuracy.py
# run_file=run_c3d_classify.py
# run_file=run_finetune_c3d.py
# run_file=run_feature_extract_mica.py
# run_file=run_classify_mica_prob.py
# run_file=run_test_net_c3d_mica.py
# run_file=run_classifier_prob.py
# run_file=run_classify_late_fusion.py
# run_file=AnalyseData.py
run_file=run_create_of_image.py
# run_file=run_c3d_compute_mean.py

report_folder=report
tmp_folder=Asset/tmp
asset_folder=Asset

mkdir -p $report_folder/report_$name
echo "$now start $name" >> $report_folder/time_$name.txt
echo "$now start $name" >> $report_folder/log_$name.txt
echo "$now start $name" >> $report_folder/err_$name.txt
python $run_file $name >>$report_folder/log_$name.txt 2>> $report_folder/err_$name.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop $name" >> $report_folder/time_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to $name" >> $report_folder/time_$name.txt
mv *$name* $report_folder/*$name* $report_folder/report_$name
cp $tmp_folder/input.txt $tmp_folder/output.txt $tmp_folder/model.prototxt $tmp_folder/solver.prototxt $report_folder/report_$name
# cp $asset_folder/mica_train.txt $asset_folder/mica_test.txt $report_folder/report_$name
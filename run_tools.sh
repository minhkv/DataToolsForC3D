STARTTIME=$(date +%s)
now=$(date)
name="feature_extraction_lost_ucf_2"
# name="classify_ucf_2_prob"

run_file=run_c3d_feature_extraction.py
# run_file=run_c3d_convert_bin_to_csv.py
# run_file=run_c3d_classify.py
# run_file=run_classifier_prob.py

mkdir -p report_$name
echo "$now start $name" >> time_$name.txt
echo "$now start $name" >> log_$name.txt
echo "$now start $name" >> err_$name.txt
python $run_file $name >>log_$name.txt 2>> err_$name.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop $name" >> time_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to $name" >> time_$name.txt
mv *$name.* report_$name
STARTTIME=$(date +%s)
now=$(date)
name="extract_feature_ucf_split_3"
mkdir -p report_$name
echo "$now start extracting feature" >> time_$name.txt
echo "$now start extracting feature" >> log_$name.txt
echo "$now start extracting feature" >> err_$name.txt
python run_c3d_feature_extraction.py >>log.txt 2>> err_$name.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop extracting feature" >> time_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to extract feature" >> time_$name.txt
mv *$name.txt report_$name
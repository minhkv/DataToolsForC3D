STARTTIME=$(date +%s)
now=$(date)
echo "$now start extracting feature" >> time.txt
echo "$now start extracting feature" >> log.txt
echo "$now start extracting feature" >> err.txt
python run_c3d_feature_extraction.py >>log.txt 2>> err.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop extracting feature" >> time.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to extract feature" >> time.txt
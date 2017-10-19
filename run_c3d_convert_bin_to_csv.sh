STARTTIME=$(date +%s)
now=$(date)
echo "$now start converting feature" >> time.txt
echo "$now start converting feature" >> log.txt
echo "$now start converting feature" >> err.txt
python run_c3d_convert_bin_to_csv.py >>log.txt 2>> err.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop converting feature" >> time.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to convert feature" >> time.txt
STARTTIME=$(date +%s)
now=$(date)
echo "$now start converting feature" >> time_sport1m.txt
echo "$now start converting feature" >> log_sport1m.txt
echo "$now start converting feature" >> err_sport1m.txt
python run_c3d_convert_bin_to_csv.py >>log_sport1m.txt 2>> err_sport1m.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop converting feature" >> time_sport1m.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to convert feature" >> time_sport1m.txt
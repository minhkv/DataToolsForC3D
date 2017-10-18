STARTTIME=$(date +%s)
now=$(date)
echo "$now start testing" >> time.txt
echo "$now start testing" >> log.txt
echo "$now start testing" >> err.txt
python run_c3d_test_net.py >>log.txt 2>> err.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop testing" >> time.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to test" >> time.txt
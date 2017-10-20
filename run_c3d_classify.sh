STARTTIME=$(date +%s)
now=$(date)
echo "$now start classification" >> time_classify_sport1m_split_1.txt
echo "$now start classification" >> log_classify_sport1m_split_1.txt
echo "$now start classification" >> err_classify_sport1m_split_1.txt
python run_c3d_classify.py >>log_classify_sport1m_split_1.txt 2>> err_classify_sport1m_split_1.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop classification" >> time_classify_sport1m_split_1.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to classify" >> time_classify_sport1m_split_1.txt
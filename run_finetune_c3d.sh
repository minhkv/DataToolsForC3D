STARTTIME=$(date +%s)
now=$(date)
echo "$now start finetuning" >> time.txt
echo "$now start finetuning" >> log.txt
echo "$now start finetuning" >> err.txt
python run_finetune_c3d.py >>log.txt 2>> err.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop finetuning" >> time.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to finetune" >> time.txt
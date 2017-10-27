STARTTIME=$(date +%s)
now=$(date)
name="finetune_ucf_2"
run_file=run_finetune_c3d.py
mkdir -p report_$name
echo "$now start finetuning" >> time_$name.txt
echo "$now start finetuning" >> log_$name.txt
echo "$now start finetuning" >> err_$name.txt
python $run_file >>log_$name.txt 2>> err.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop finetuning" >> time_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to finetune" >> time_$name.txt
mv *$name.* report_$name
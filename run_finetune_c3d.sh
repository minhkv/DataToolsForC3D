STARTTIME=$(date +%s)
now=$(date)
name="finetune_ucf_2_demo"
run_file=run_finetune_c3d.py
mkdir -p report_$name
echo "$now start $name" >> time_$name.txt
echo "$now start $name" >> log_$name.txt
echo "$now start $name" >> err_$name.txt
python $run_file >>log_$name.txt 2>> err_$name.txt
ENDTIME=$(date +%s)
now=$(date)
echo "$now stop $name" >> time_$name.txt
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to $name" >> time_$name.txt
mv *$name.* report_$name
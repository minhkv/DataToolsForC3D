STARTTIME=$(date +%s)
now=$(date)
# name="feature_extraction_lost_sport1m"
# name="convert_lost_ucf_2_fc6"
# name="test_net_finetune_ucf_3"
name="classify_sport1m_1_fc6_bin"

run_file=AnalyseData.py

report_folder=report

python $run_file $name

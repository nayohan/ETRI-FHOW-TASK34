CUDA_VISIBLE_DEVICES="0" python ./main.py --mode train \
                                     --in_file_trn_dialog ../data/task4/ddata.wst.txt.2021.6.9 \
                                     --in_file_fashion ../data/task4/mdata.wst.txt.2021.10.18 \
                                     --in_dir_img_feats ../data/task4/img_feats \
                                     --subWordEmb_path ../data/task4/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                     --model_path ./gAIa_model \
                                     --mem_size 32 \
                                     --key_size 300 \
                                     --hops 3 \
                                     --eval_node [300,4000,4000] \
                                     --tf_dropout 0.3 \
                                     --tf_nhead 4 \
                                     --tf_ff_dim 4096 \
                                     --tf_num_layers 4 \
                                     --epochs 100 \
                                     --save_freq 10 \
                                     --batch_size 10 \
                                     --learning_rate 0.001 \
                                     --max_grad_norm 20.0 \
                                     --use_multimodal True \
                                     --use_dropout True \
                                     --eval_zero_prob 0.5 \
                                     --corr_thres 0.9 \



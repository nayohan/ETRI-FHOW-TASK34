CUDA_VISIBLE_DEVICES="0" python3 ./main.py --mode zsl \
                                   --in_file_tst_dialog ../data/task4/fs_eval_t1.wst.dev \
                                   --in_file_fashion ../data/task4/mdata.wst.txt.2021.10.18 \
                                   --in_dir_img_feats ../data/task4/img_feats \
                                   --subWordEmb_path ../data/task4/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                   --model_path ./gAIa_model \
                                   --model_file gAIa-500.pt \
                                   --mem_size 16 \
                                   --key_size 300 \
                                   --hops 3 \
                                   --eval_node [600,4000,4000] \
                                   --tf_nhead 4 \
                                   --tf_ff_dim 4096 \
                                   --tf_num_layers 4 \
                                   --batch_size 100 \
                                   --use_multimodal True \





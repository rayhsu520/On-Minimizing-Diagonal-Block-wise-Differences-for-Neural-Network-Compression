python main.py -lmd ../model/deepc/model_quantized_retrain10.ptmodel -lmq ../model/fc_q/model_quantized_retrain10.ptmodel -lmm ../model/mesa/checkpoint_quantized_re_alpha_0.0_10.tar -lmi ../model/fc/model_initial_end.ptmodel -lmp ../model/mpd/checkpoint_initial_p_alpha_0.0_100.tar -p fc1=8,fc2=8,fc3=10 -b conv=8,fc=5 -tm d -sd ../output_compression


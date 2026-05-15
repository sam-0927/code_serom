# Training
python train_stage1.py --task_name stage1
학습이 끝나면 outputs/stage1 생성.

configs/config_stage2.yaml 에 stage1_ckpt_path 에 stage1 에서 학습한 모델 주소 입력. model_030.tar 같은거
python train_stage2.py --task_name stage2
학습이 끝나면 outputs/stage2 생성.

# Inference
python infer_voc.py --ckpt outputs/ft_w_cnn/stage2_wo_voc_pre/checkpoints/best_mel_model.tar --config outputs/ft_w_cnn/stage2_wo_voc_pre/config_stage2.yaml --output_dir serom

# Pretrained model
clean_vocoder: clean speech로 vocos 학습한거. (사용하진 않았음. ablation용으로 학습했으나 한번도 안씀)
stage1_model: stage2에 사용된 모델. epoch 30.
stage2_model: 최종 모델.

stage1_wo_ctc_model1, stage2_wo_ctc_model: ctc loss 를 사용하지 않고 학습한 모델. 구조는 ctc fc layer 빼고, 1024 -> 1024 인 fc layer 만 사용.
stage1_wo_trans_model, stage2_wo_trans_model: transformer (decoder)를 사용하지 않고 학습한 모델. acoustic emb + content emb -> fc(1024 -> 80) -> mel

https://drive.google.com/drive/u/0/folders/1N7klUlLo0Oyox_o1Rzukm_rY-YwgwUX1


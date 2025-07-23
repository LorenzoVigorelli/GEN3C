#!/usr/bin/env bash

echo "Start running for first trajectory"

CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/leftcentered/ \
  --video_save_name md03g1 \
  --guidance 1 \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models

CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/leftcentered/ \
  --video_save_name md04g3 \
  --movement_distance 0.4 \
  --guidance 1 \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models

CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/leftcentered/ \
  --video_save_name md05g1 \
  --movement_distance 0.5 \
  --guidance 1 \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models

# md06g1, md07g1, md08g1 analogous a md05g1, solo movement_distance diverso
for dist in 0.6 0.7 0.8; do
  name="md0${dist#0}g1"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/leftcentered/ \
    --video_save_name $name \
    --movement_distance $dist \
    --guidance 1 \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done

# md03g2 e md03g3: senza movement_distance
for guide in 2 3; do
  name="md03g${guide}"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/leftcentered/ \
    --video_save_name $name \
    --guidance $guide \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done


echo "Start running for second trajectory"

# trajectory right + camera_rotation no_rotation
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/rightfixed/ \
  --video_save_name md03g1 \
  --guidance 1 \
  --trajectory right \
  --camera_rotation no_rotation \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models

for dist in 0.4 0.5 0.6 0.7 0.8; do
  name="md0${dist#0}g1"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/rightfixed/ \
    --video_save_name $name \
    --movement_distance $dist \
    --guidance 1 \
    --trajectory right \
    --camera_rotation no_rotation \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done

# md03g2 e md03g3 in second trajectory
for guide in 2 3; do
  name="md03g${guide}"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/rightfixed/ \
    --video_save_name $name \
    --guidance $guide \
    --trajectory right \
    --camera_rotation no_rotation \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done


echo "Start running for third trajectory"

# trajectory right, no movement for md03g1 e md03g3
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/rightcentered/ \
  --video_save_name md03g1 \
  --guidance 1 \
  --trajectory right \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models

for dist in 0.4 0.5 0.6; do
  # md04g3 ha movement_distance ma guid 1
  name="md0${dist#0}g3"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/rightcentered/ \
    --video_save_name $name \
    --movement_distance $dist \
    --guidance 1 \
    --trajectory right \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done

# correzione typo "rigth" → "right"
# md03g3 al termine
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/rightcentered/ \
  --video_save_name md03g3 \
  --guidance 3 \
  --trajectory right \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models


echo "Start running for fourth trajectory"

for dist in 1 0.4 0.5 0.6 0.7 0.8; do
  # usa md03g1/md04g3...md08g1 per leftfixed
  name="md0${dist}g1"
  extra=""
  [[ $dist != 1 ]] && extra="--movement_distance $dist"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/leftfixed/ \
    --video_save_name $name \
    --guidance 1 \
    $extra \
    --camera_rotation no_rotation \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done

# md03g2 e md03g3 per fourth trajectory
for guide in 2 3; do
  name="md03g${guide}"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/leftfixed/ \
    --video_save_name $name \
    --guidance $guide \
    --camera_rotation no_rotation \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done

#!/usr/bin/env bash

echo "Start running for fifth trajectory (clockwise)"

# md03g1 (default movement)
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
  --checkpoint_dir checkpoints \
  --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
  --video_save_folder buffer/upcentered/ \
  --video_save_name md03g1 \
  --guidance 1 \
  --trajectory up \
  --save_buffer \
  --offload_text_encoder_model \
  --offload_tokenizer \
  --offload_prompt_upsampler \
  --offload_guardrail_models

# md04g1...md08g1 con movement_distance
for dist in 0.4 0.5 0.6 0.7 0.8; do
  name="md0${dist#0}g1"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/upcentered/ \
    --video_save_name $name \
    --movement_distance $dist \
    --guidance 1 \
    --trajectory up \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done

# md03g2 e md03g3 (default movement)
for guide in 2 3; do
  name="md03g${guide}"
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -m cosmos_predict1.diffusion.inference.workload.renderingCache \
    --checkpoint_dir checkpoints \
    --input_image_path /export/scratch/lvigorel/GEN3C/assets/diffusion/000.png \
    --video_save_folder buffer/upcentered/ \
    --video_save_name $name \
    --guidance $guide \
    --trajectory up \
    --save_buffer \
    --offload_text_encoder_model \
    --offload_tokenizer \
    --offload_prompt_upsampler \
    --offload_guardrail_models
done



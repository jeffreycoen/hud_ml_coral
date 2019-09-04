# hud_ml_coral

export MODELS="$HOME/hud_ml_coral/edgetpuvision-master/edgetpuvision/all_models"

python3 -m edgetpuvision.detect \
    --model ${MODELS}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --labels ${MODELS}/coco_labels.txt \
    --top_k 3 \
    --threshold 0.5 \
    --filter person \
    --color white

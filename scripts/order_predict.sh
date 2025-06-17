python3 ../VORO/src/main.py \
    --te_data ../output/order_results/predict.json\
    --model_path ../models/order_model/inference.pth\
    --out_folder ../output/order_results/\
    --eval_only True\
    --model mobilenetv3\
    --image_path ./saved/\
    --image_flag True\
    --batch_size 200\
    --gpu 0

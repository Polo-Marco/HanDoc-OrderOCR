python3 ../VORO/src/main.py --tr_data ../output/order_results/predict.json\
    --val_data ../output/order_results/predict.json\
    --te_data ../output/order_results/predict.json\
    --model_path ../models/order_model/inference.pth\
    --out_folder ../output/order_results/\
    --eval_only True\
    --model mobilenetv3\
    --image_path ./saved/\
    --image_flag True\
    --gpu 0

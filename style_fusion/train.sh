python train.py \
  --save_dir ./experiments/DWTFusionNet_content_size=1024_mseloss \
  --loss mse \
  --batch_size 32 \
  --gpu 6 \
  --max_iter 50000 \
  --save_model_interval 5000


  python train.py \
  --save_dir ./experiments/DWTFusionNet_content_size=1024_l1loss \
  --loss l1 \
  --batch_size 4 \
  --gpu 7 \
  --max_iter 100000 \
  --save_model_interval 5000

python train_all_loss.py \
  --save_dir ./experiments/DWTFusionNet_content_size=1024_l1+all_loss2 \
  --loss l1 \
  --batch_size 4 \
  --gpu 6 \
  --max_iter 100000 \
  --save_model_interval 5000


python train.py \
  --data_root /data/yueyisui/datasets/DIOR/JPEGImages-2048-256_fusion \
  --save_dir ./experiments/DWTFusionNet_content_size=2048_l1loss \
  --content_size 2048 \
  --loss l1 \
  --batch_size 4 \
  --gpu 5 \
  --max_iter 100000 \
  --save_model_interval 5000

python train_all_loss.py \
  --save_dir ./experiments/FusionNet_content_size=1024_l1+all_loss \
  --loss l1 \
  --batch_size 4 \
  --gpu 6 \
  --max_iter 100000 \
  --save_model_interval 5000

python train_all_loss.py \
  --save_dir ./experiments/DWTFusionNet_content_size=2048_l1+all_loss \
  --loss l1 \
  --batch_size 4 \
  --gpu 2 \
  --max_iter 100000 \
  --save_model_interval 5000

  python train_all_loss.py \
  --save_dir ./experiments/DWTFusionNet_content_size=2048_l1+all_loss_test \
  --loss l1 \
  --batch_size 4 \
  --gpu 1 \
  --max_iter 200000 \
  --save_model_interval 5000
export CUDA_VISIBLE_DEVICES=0
python main.py data/Cameron/ --workspace trial/Cameron/ -O --iters 70000 \
    --data_range 0 -1 \
    --dim_eye 4 \
    --lr 0.005 --lr_net 0.0005 \
    --num_rays 65536 \
    --patch_size 32 

python main_sr.py data/Cameron/ --workspace trial/Cameron/ -O --iters 150000 \
    --data_range 0 -1 \
    --dim_eye 4 \
    --patch_size 32 \
    --srtask \
    --num_rays 16384 \
    --lr 0.005 \
    --lr_net 0.0005 \
    --weight_pcp 0.05 \
    --weight_style 0.01 \
    --weight_gan 0.01 --test_tile 450\ 
    #--ftsr_path   'trial/Cameron/modelsr_ckpt/sresrnet_17.pth' 



# python main.py data/Sunak/ --workspace trial/Sunak/ -O --iters 70000 \
#     --data_range 0 -1 \
#     --dim_eye 6 \
#     --lr 0.005 --lr_net 0.0005 \
#     --num_rays 65536 \
#     --patch_size 32 # --num_rays 261900

# python main_sr.py data/Sunak/ --workspace trial/Sunak/ -O --iters 150000 \
#     --data_range 0 -1 \
#     --dim_eye 6 \
#     --patch_size 32 \
#     --srtask \
#     --num_rays 16384 \
#     --lr 0.005 \
#     --lr_net 0.0005 \
#     --weight_pcp 0.05 \
#     --weight_style 0.01 \
#     --weight_gan 0.01 --test_tile 450\
#     #--ftsr_path   'trial/Cameron1/modelsr_ckpt/sresrnet_17.pth' #--aud 'data/auds/Trudeau.npy'









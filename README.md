This is mainly base on mmdetection
screen -dmS xxx
screen -ls
screen -r xxx

python tools/train.py configs/vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py --gpu-ids 0 
./tools/dist_train.sh configs/vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py 2

python tools/test.py configs/vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py last/epoch_25.pth --format-only --options jsonfile_prefix=./$file_name


python = 3.7.10
pytorch = 1.7.0
cudatoolkit = 11.0.221
torchvision = 0.8.0
gcc = 5.4.0

NVIDIA-SMI 455.23.04    Driver Version: 455.23.04    CUDA Version: 11.1 

最终排名
B榜：42名，精度0.50539070

baseline：VFNet
新增Retinex,Mixup,MotionBlur等数据增强。
包含各种脚本：统计图片信息和annotations尺度、划分数据集、coco2voc、voc2coco、Retinex、txt_output、json操作等等。
可使用伪标签去刷点，soft_nms阈值可以设置0.5/0.7

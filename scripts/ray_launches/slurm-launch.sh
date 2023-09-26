export i=wikihow_chatGPT

echo $VAR

python slurm-launch.py --exp-name test --num-nodes 2 --num-gpus 4 --partition develbooster

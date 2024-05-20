TRAIN_CONFIGS='configs/v1.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.gpu)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}
#CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m romp.train --configs_yml=${TRAIN_CONFIGS} > 'log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &
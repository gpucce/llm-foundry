salloc --account IscrC_GELATINO \
    --partition boost_usr_prod \
    --exclude lrdn[0809] \
    --time 00:30:00 \
    --gres gpu:2 \
    --nodes 4 \
    --mem 250G \
    --ntasks-per-node 2 \
    --cpus-per-task 8 \
    --wait-all-nodes 1 \
    /bin/bash
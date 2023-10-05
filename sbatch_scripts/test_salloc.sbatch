salloc --account IscrC_GELATINO \
    --partition boost_usr_prod \
    --time 00:30:00 \
    --gres gpu:2 \
    --nodes 1 \
    --ntasks-per-node 2 \
    --cpus-per-task 8 \
    /bin/bash
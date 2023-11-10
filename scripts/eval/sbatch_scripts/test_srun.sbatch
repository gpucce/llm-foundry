srun --pty --account IscrC_GELATINO \
    --partition boost_usr_prod \
    --time 00:30:00 \
    --qos boost_qos_dbg \
    --gres gpu:4 \
    --nodes 1 \
    --ntasks-per-node 1 \
    --cpus-per-task 32 \
    /bin/bash
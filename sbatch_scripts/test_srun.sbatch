srun --pty --account IscrC_GELATINO \
    --partition boost_usr_prod \
    --time 00:30:00 \
    --gres gpu:4 \
    --qos boost_qos_dbg \
    --nodes 1 \
    --ntasks-per-node 1 \
    --cpus-per-task 32 \
    /bin/bash

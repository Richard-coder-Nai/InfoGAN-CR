i=0
N=10
for info in 0 1.0 2.0 4.0 8.0
do
  for cr in 0 0.1 0.5 1.0 2.0 3.0 5.0
  do
    for seed in 0 1 2 3 4 5
    do
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)
    export TF_FORCE_GPU_ALLOW_GROWTH=true
      de=$info
    ((i=i%N));((i++==0)) && wait
    dir_name=infogan_cr_de${de}_info${info}_cr${cr}_seed$seed
    if [ -d $dir_name/checkpoint/global_id-299999 ]
    then
      continue
    else
      rm -r $dir_name
    fi
    mkdir -p ./$dir_name
    python infogan_cr.py \
      --dir_name $dir_name \
      --gin_bindings "gan.info_coe_de = $de"  \
      "gan.info_coe_infod = $info" \
      "gan.cr_coe_increase = $cr" \
      "random_seed.seed = $seed" \
       1>./$dir_name/run.log  2>&1  &
    echo $dir_name
    sleep 20s
  done
done
done


i=0
N=4
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)
export TF_FORCE_GPU_ALLOW_GROWTH=true
((i=i%N));((i++==0)) && wait
for cr_batch in 25000 50000
do
  for gap_batch in 85000 40000
  do
    dir_name=cr_batch${cr_batch}_gap_batch${gap_batch}
    mkdir -p ./$dir_name
    python infogan_cr.py \
      --dir_name $dir_name \
      --gin_bindings "gan.info_coe_de = 2.0"  \
      "gan.info_coe_infod = 2.0" \
      "gan.cr_coe_increase_batch = $cr_batch" \
      "gan.gap_decrease_batch = $gap_batch" \
      1>./$dir_name/run.log  2>&1  &
    echo $dir_name
    sleep 20s
  done
done



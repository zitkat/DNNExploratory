#!/bin/bash
#PBS -q gpu
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=30gb:cluster=adan
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/
#PBS -m ae
trap 'clean_scratch' TERM EXIT

# -- envioronment
# Use Singularity image:
# tbt_torch_21.03-py3.sif

# -- tested by:
##$ qsub -i -l select=1:ncpus=1:ngpus=1:cluster=adan:mem=10gb:scratch_ssd=30gb -l walltime=3:00:00 -q gpu
##$ mkdir $scratchdir/dataset_h5
##$ cp -r /storage/plzen4-ntis/projects/korpusy_cv/clef2021_medical_tuberculosistbt/dataset_h5v0/* $scratchdir/dataset_h5
##$ cp /storage/plzen4-ntis/projects/korpusy_cv/clef2021_medical_tuberculosistbt/environments/tbt_torch_21.03-py3.sif $scratchdir

##$ cd /storage/plzen1/home/zitkat/tuberculosistbt/devel/zita

##$ singularity shell --nv -B $SCRATCHDIR $SCRATCHDIR/tbt_torch_21.03-py3.sif
##singularity>  python train_2D.py --name train_2D_test_mem_2 -d "$SCRATCHDIDR/dataset_h5" -f 1 --debug

##$ singularity exec --nv -B "$SCRATCHDIR" "$SCRATCHDIR"/tbt_torch_21.03-py3.sif python train_2D.py
#       --name "$name"
#       -d "$SCRATCHDIR"/dataset_h5
#       -f 1
#       --debug
#       &>./trainlog"$today".txt


# -- actual run
name=train_2D_h5_2

mkdir "$SCRATCHDIR"/dataset_h5 || exit $LINENO
cp -r /storage/plzen4-ntis/projects/korpusy_cv/CLEF2021_medical_TuberculosisTBT/dataset_h5v0/* "$SCRATCHDIR"/dataset_h5 || exit $LINENO
cp /storage/plzen4-ntis/projects/korpusy_cv/CLEF2021_medical_TuberculosisTBT/environments/tbt_torch_21.03-py3.sif "$SCRATCHDIR" || exit $LINENO

cd /storage/plzen1/home/zitkat/TuberculosisTBT/devel/Zita || exit $LINENO
today=$(date +%Y%m%d%H%M)
singularity exec --nv -B "$SCRATCHDIR"  "$SCRATCHDIR"/tbt_torch_21.03-py3.sif python train_2D.py \
  --name "$name" \
  -d "$SCRATCHDIR"/dataset_h5 \
  -f 1 \
  --nepoch 20 \
  --lr 0.00002 \
  --tf_log \
  --save_epoch_freq 5 \
  &>./trainlog"$today".txt

# In case of local quota issues set these
#export SINGULARITY_CACHEDIR=/storage/plzen1/home/zitkat/sing_cache
#export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
#export SINGULARITY_TMPDIR=""

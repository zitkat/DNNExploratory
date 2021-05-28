#!/bin/bash
#PBS -q gpu
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=20gb:cluster=adan
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/
#PBS -m ae


# -- environment was set up as:
# source activate /storage/plzen1/home/zitkat/.conda/envs/Clef_TBT_env
# python -m


# -- tested by:
# // commands used in interactive mode to test script start
## qsub -I -l select=1:ncpus=1:ngpus=1:scratch_ssd=20gb -l walltime=1:00:00 -q gpu


# -- actual run
trap 'clean_scratch' TERM EXIT

module add anaconda3-2019.10
source activate /storage/plzen1/home/zitkat/.conda/envs/Clef_TBT_env || exit $LINENO

mkdir "$SCRATCHDIR"/data || exit $LINENO
DATA_PATH=$SCRATCHDIR/data
cp -r /storage/plzen1/home/zitkat/DNNExploratory/data "$SCRATCHDIR" || exit $LINENO

cd /storage/plzen1/home/zitkat/DNNExploratory || exit $LINENO
today=$(date +%Y%m%d%H%M)
python render_timm_model.py seresnext50_32x4d -w pretrained -sv v1 \
                            --output "$DATA_PATH" \
                            --hide-progress \
                            --settings-file "$DATA_PATH"/settings.ods > today.log
cp -r "$DATA_PATH" /storage/plzen1/home/zitkat/DNNExplorer/data || exit $LINENO



#!/bin/bash
#PBS -q gpu
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=20gb:cluster=adan
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/
#PBS -m ae


# -- environment was set up as:
# source activate /storage/plzen1/home/zitkat/.conda/envs/Clef_TBT_env
# python -m


# -- tested by:
# // commands used in interactive mode to test script start
## qsub -I -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=20gb -l walltime=1:00:00 -q gpu


# -- actual run
trap 'clean_scratch' TERM EXIT

module add anaconda3-2019.10
source activate Clef_TBT_env || exit $LINENO

#mkdir "$SCRATCHDIR"/data || exit $LINENO
cp -r /storage/plzen1/home/zitkat/DNNExploratory/ "$SCRATCHDIR" || exit $LINENO
DATA_PATH=$SCRATCHDIR/DNNExploratory/data
WORK_PATH=$SCRATCHDIR/DNNExploratory/

cd "$WORK_PATH" || exit $LINENO
today=$(date +%Y%m%d%H%M)
python render_timm_model.py seresnext50_32x4d -w initialized  -sv v1 \
                            --output "$DATA_PATH" \
                            --hide-progress \
                            --settings-file "$DATA_PATH"/settings.ods > "$DATA_PATH"/"$today".log
cp -ru "$DATA_PATH" /storage/plzen1/home/zitkat/DNNExploratoryRes/ || export CLEAN_SCRATCH=False




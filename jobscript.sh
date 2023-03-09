# PBS -lwalltime=20:00:00
# PBS -lselect=1:ncpus=1:mem=2gb
module load anaconda3/personal
source activate py39
python $HOME/FYPDeepDeconstructedLearning/main.py
mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID

#!/bin/sh
for i in 059
do
    echo "#!/bin/sh">>job.sbatch
    echo "#SBATCH --export=NONE">>job.sbatch
    echo "#SBATCH -A nssac_students">>job.sbatch
    echo "#SBATCH --time=12:00:00">>job.sbatch
    echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>job.sbatch
    echo "#SBATCH --mem=200g">>job.sbatch
    echo "#SBATCH -p bii">>job.sbatch
    echo "#SBATCH -J synthetic-distribution">>job.sbatch
    echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/secjob$i.out -e /scratch/rm5nz/synthetic-distribution/secjob$i.err">>job.sbatch
    echo "module load gurobi">>job.sbatch
    echo "module load anaconda/2019.10-py3.7">>job.sbatch
    echo "source activate rounak">>job.sbatch
    echo "python distnw-sbatch-secnet-skip.py $i">>job.sbatch
    sbatch job.sbatch
    rm job.sbatch
done
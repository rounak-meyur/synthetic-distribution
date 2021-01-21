#!/bin/sh
for i in 121143 121144 147793 148717 148718 148719 148720 148721 148723 150353 150589 150638 150692 150722 150723 150724 150725 150726 150727 150728
do
    echo "#!/bin/sh">>distjob.sbatch
    echo "#SBATCH --export=NONE">>distjob.sbatch
    echo "#SBATCH -A nssac_students">>distjob.sbatch
    echo "#SBATCH --time=12:00:00">>distjob.sbatch
    echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>distjob.sbatch
    echo "#SBATCH --mem=250g">>distjob.sbatch
    echo "#SBATCH -p bii">>distjob.sbatch
    echo "#SBATCH -J synthetic-distribution">>distjob.sbatch
    echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/distnet$i.out -e /scratch/rm5nz/synthetic-distribution/distnet$i.err">>distjob.sbatch
    echo "module load anaconda/2019.10-py3.7">>distjob.sbatch
    echo "source activate rounak">>distjob.sbatch
    echo "module load gurobi">>distjob.sbatch
    echo "python distnw-sbatch-powerflow.py $i">>distjob.sbatch
    sbatch distjob.sbatch
    rm distjob.sbatch
done
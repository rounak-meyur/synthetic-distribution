#!/bin/sh
for i in 121143 121144 147793 148717 148718 148719 148720 148721 148723 150353 150589 150638 150692 150722 150723 150724 150725 150726 150727 150728
do
    echo "#!/bin/sh">>primjob.sbatch
    echo "#SBATCH --export=NONE">>primjob.sbatch
    echo "#SBATCH -A nssac_students">>primjob.sbatch
    echo "#SBATCH --time=24:00:00">>primjob.sbatch
    echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>primjob.sbatch
    echo "#SBATCH --mem=150g">>primjob.sbatch
    echo "#SBATCH -p bii">>primjob.sbatch
    echo "#SBATCH -J synthetic-distribution">>primjob.sbatch
    echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/primnet$i.out -e /scratch/rm5nz/synthetic-distribution/primnet$i.err">>primjob.sbatch
    echo "module load anaconda/2019.10-py3.7">>primjob.sbatch
    echo "source activate rounak">>primjob.sbatch
    echo "module load gurobi">>primjob.sbatch
    echo "python distnw-sbatch-primnet.py $i">>primjob.sbatch
    sbatch primjob.sbatch
    rm primjob.sbatch
done
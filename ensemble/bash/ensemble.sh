#!/bin/sh
for i in 150723 150724
do
    echo "#!/bin/sh">>ensjob.sbatch
    echo "#SBATCH --export=NONE">>ensjob.sbatch
    echo "#SBATCH -A nssac_students">>ensjob.sbatch
    echo "#SBATCH --time=2:00:00">>ensjob.sbatch
    echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>ensjob.sbatch
    echo "#SBATCH --mem=250g">>ensjob.sbatch
    echo "#SBATCH -p bii">>ensjob.sbatch
    echo "#SBATCH -J synthetic-distribution">>ensjob.sbatch
    echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/ensemble$i.out -e /scratch/rm5nz/synthetic-distribution/ensemble$i.err">>ensjob.sbatch
    echo "module load anaconda/2019.10-py3.7">>ensjob.sbatch
    echo "source activate rounak">>ensjob.sbatch
    echo "python distnw-sbatch-ensemble.py $i">>ensjob.sbatch
    sbatch ensjob.sbatch
    rm ensjob.sbatch
done
#!/bin/sh
for i in 121144 147793
do
    for (( j = 1; j <= 4; j++ ))
    do
        for (( k = j+1; k <= 4; k++ ))
        do
            echo "#!/bin/sh">>compjob.sbatch
            echo "#SBATCH --export=NONE">>compjob.sbatch
            echo "#SBATCH -A nssac_students">>compjob.sbatch
            echo "#SBATCH --time=10:00:00">>compjob.sbatch
            echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>compjob.sbatch
            echo "#SBATCH --mem=200g">>compjob.sbatch
            echo "#SBATCH -p bii">>compjob.sbatch
            echo "#SBATCH -J synthetic-distribution">>compjob.sbatch
            echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/compnet${i}_ens${j}_ens${k}.out -e /scratch/rm5nz/synthetic-distribution/compnet${i}_ens${j}_ens${k}.err">>compjob.sbatch
            echo "module load anaconda/2019.10-py3.7">>compjob.sbatch
            echo "source activate rounak">>compjob.sbatch
            echo "python sbatch-compare-hauss.py $i $j $k">>compjob.sbatch
            sbatch compjob.sbatch
            rm compjob.sbatch
        done
    done
done
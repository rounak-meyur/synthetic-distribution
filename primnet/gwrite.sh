#!/bin/sh
echo "#!/bin/sh">>gjob.sbatch
echo "#SBATCH --export=NONE">>gjob.sbatch
echo "#SBATCH -A nssac_students">>gjob.sbatch
echo "#SBATCH --time=10:00:00">>gjob.sbatch
echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>gjob.sbatch
echo "#SBATCH --mem=100g">>gjob.sbatch
echo "#SBATCH -p bii">>gjob.sbatch
echo "#SBATCH -J synthetic-distribution">>gjob.sbatch
echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/gwrite.out -e /scratch/rm5nz/synthetic-distribution/gwrite.err">>gjob.sbatch
echo "module load anaconda/2019.10-py3.7">>gjob.sbatch
echo "source activate rounak">>gjob.sbatch
echo "python distnw-sbatch-gwrite.py">>gjob.sbatch
sbatch gjob.sbatch
rm gjob.sbatch
#!/bin/sh
echo "#!/bin/sh">>vjob.sbatch
echo "#SBATCH --export=NONE">>vjob.sbatch
echo "#SBATCH -A nssac_students">>vjob.sbatch
echo "#SBATCH --time=10:00:00">>vjob.sbatch
echo "#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1">>vjob.sbatch
echo "#SBATCH --mem=100g">>vjob.sbatch
echo "#SBATCH -p bii">>vjob.sbatch
echo "#SBATCH -J synthetic-distribution">>vjob.sbatch
echo "#SBATCH -o /scratch/rm5nz/synthetic-distribution/gwrite.out -e /scratch/rm5nz/synthetic-distribution/gwrite.err">>vjob.sbatch
echo "module load anaconda/2019.10-py3.7">>vjob.sbatch
echo "source activate rounak">>vjob.sbatch
echo "python distnw-sbatch-voronoi.py">>vjob.sbatch
sbatch vjob.sbatch
rm vjob.sbatch
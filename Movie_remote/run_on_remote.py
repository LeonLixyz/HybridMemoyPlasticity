import os
import subprocess
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
Local_download_path = '/Users/leon66/Desktop/Neuroscience/Movie_remote/Result'
USERNAME = 'al4263'
REMOTE_SERVER = 'al4263@ginsburg.rcs.columbia.edu'
REMOTE_SYNC_SERVER = 'al4263@motion.rcs.columbia.edu'
REMOTE_PATH = '/burg/theory/users/al4263/Movie_remote'
REMOTE_PATH_Result = '/burg/theory/users/al4263/Movie_remote/Result'

commands = [
 'ML [100] [1] [M] 4',
 'ML [100] [0] [M] 4',
 'ML [100] [0.5] [M] 4',
 'ML [100] [1] [A] 4',
 'ML [100] [0] [A] 4',
 'ML [100] [0.5] [A] 4',
 'end'
 ]

# Write the experiment commands with underscores to a file
with open('experiments.txt', 'w') as f:
    f.write("\n".join(commands))

# Generate the SLURM submit script
submit_sh = f"""#!/bin/sh

#SBATCH --account=theory
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=5-00:00:00
#SBATCH --array=0-{len(commands)-1}%12
#SBATCH --job-name=Movie_experiments_name
#SBATCH --output=slurm/slurm_%x_%a_%A.out
#SBATCH --error=slurm/slurm_%x_%a_%A.err 

[[ ! -d slurm ]] && mkdir slurm
experiment_name=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" experiments.txt)
echo Queueing experiment $experiment_name
source ~/.bashrc
conda activate pytorch_env
srun python Main.py $experiment_name
conda deactivate
"""



status = sys.argv[1]
if status == 'download':
    rsync = f'rsync -avz --verbose {REMOTE_SYNC_SERVER}:{REMOTE_PATH_Result}/ {Local_download_path}'
    print(f'Syncing...\n {rsync}')
    subprocess.check_call(rsync, shell=True)

elif status == 'run':
    # Write the submit script to a file
    with open('submit.sh', 'w') as f:
        f.write(submit_sh)

    ssh_mkdir = f'ssh {REMOTE_SERVER} "mkdir -p {REMOTE_PATH}"'
    print(f'Creating remote directory...\n {ssh_mkdir}')
    subprocess.check_call(ssh_mkdir, shell=True)

    # Sync the necessary files to the remote server
    rsync = f'rsync -avz --verbose {LOCAL_PATH}/ {REMOTE_SYNC_SERVER}:{REMOTE_PATH}'
    print(f'Syncing...\n {rsync}')
    subprocess.check_call(rsync, shell=True)

    # Run the submit script on the remote server
    ssh_sbatch = f'ssh {REMOTE_SERVER} "cd {REMOTE_PATH}; sbatch submit.sh; squeue -u {USERNAME}"'
    print(f'Batching...\n {ssh_sbatch}')
    subprocess.check_call(ssh_sbatch, shell=True)

elif status == 'check':
    subprocess.check_call(f'ssh {REMOTE_SERVER} "squeue -u {USERNAME}"', shell=True)

elif status == 'cancel':
    subprocess.check_call(f'ssh {REMOTE_SERVER} "scancel -u {USERNAME}"', shell=True)


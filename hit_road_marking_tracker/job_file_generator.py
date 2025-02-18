import os

def slurmGenerator(filename, model, dir):
    content = f"""#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=job_{filename[11:-4]}
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output="log/out_{filename[11:-4]}.%j"
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

### Optinals ###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e-jalilifar@tti.tamu.edu

### Executables ###
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml Ultralytics/8.3.23-CUDA-11.7.0
ml Shapely/1.8.2

cd /scratch/user/ehsanjalilifar/tti/YOLOv8
python car_hit_line_detection.py --model "{model}.pt" --source "{filename[11:-4]}.mkv" --lanes ntta_entry_round2_lanes --markings ntta_entry_round2_lines --rois ROIs_close_view2
"""
    
    file_path = os.path.join(dir, f"job_{filename[11:-4]}.slurm")
    
    with open(file_path, "w", newline="\n") as f:
        f.write(content)

    print(f"Slurm job file created at {file_path}")

if __name__ == "__main__":
    videos_dir = os.path.join(os.getcwd(), 'input_files/Photos/Entry - Round 2/FAR')
    job_file_dir = os.path.join(os.getcwd(), 'jobs')
    for file in os.listdir(videos_dir):
        if file.endswith(".png"):
            slurmGenerator(file, 'yolov8m-seg', job_file_dir)
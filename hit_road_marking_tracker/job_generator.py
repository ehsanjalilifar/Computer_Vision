import os

def slurmGenerator(filename, model, dir):
    content = f"""#!/bin/bash
### Resource Allocation ###
#SBATCH --job-name=CV_YOLOv8_{filename[:-4]}
#SBATCH --time=02:15:00
#SBATCH --ntasks=1
#SBATCH --mem=6G
#SBATCH --output=log/out_CV_YOLOv8_{filename[:-4]}.%j
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
python car_hit_line_detection.py --model {model} --source {filename}"""
    
    file_path = os.path.join(dir, f"{filename[:-4]}.slurm")
    
    with open(file_path, "w", newline="\n") as f:
        f.write(content)

    print(f"Slurm job file created at {file_path}")

if __name__ == "__main__":
    dir = os.path.join(os.getcwd(), 'input_files/Videos/Stream_1')
    for file in os.listdir(dir):
        path = os.path.join(os.getcwd(), 'jobs')
        slurmGenerator(file, 'yolo11m-seg.pt', path)
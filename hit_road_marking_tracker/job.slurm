#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=CV_YOLOv8_20240611_081842_8C7D       #Set the job name
#SBATCH --time=03:00:00                                 #Set the wall clock limit to 4hr
#SBATCH --ntasks=1                                      #Request 1 task
#SBATCH --mem=6G                                        #Request 4GB per node
#SBATCH --output=log/out_CV_YOLOv8_20240611_081842_8C7D.%j  #Send stdout/err to "out.[jobID]"
#SBATCH --gres=gpu:a100:1                               #Request 1 "A100" GPU per node
#SBATCH --partition=gpu #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL                                #Send email on all job events
#SBATCH --mail-user=e-jalilifar@tti.tamu.edu           #Send all emails to e-jalilifar@tti.tamu.edu

#First Executable Line
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml Ultralytics/8.3.23-CUDA-11.7.0
ml Shapely/1.8.2

cd /scratch/user/ehsanjalilifar/tti/YOLOv8
python car_hit_line_detection.py
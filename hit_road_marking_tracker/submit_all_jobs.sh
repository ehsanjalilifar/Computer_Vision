#!/bin/bash

JOB_DIR="/scratch/user/ehsanjalilifar/tti/YOLOv8/jobs"

for job_file in "$JOB_DIR"/*.slurm; do
    echo "Submitting job file: $job_file"
    sbatch "$job_file"
done

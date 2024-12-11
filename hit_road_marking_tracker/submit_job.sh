#!/bin/bash

JOB_DIR="/scratch/user/ehsanjalilifar/tti/YOLOv8/jobs"

for job in "$JOB_DIR"/*.job; do
    echo "submitting $job"
    sbatch "$job"
done
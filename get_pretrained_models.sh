#!/usr/bin/env bash

mkdir -p checkpoints
# Place final weights here:
curl -o checkpoints/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

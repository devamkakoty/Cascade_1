#!/bin/bash
# ── Deploy Cascade Predict to AWS EC2 ────────────────────────────
# Run this ON the EC2 instance after SSH-ing in.
#
# Prerequisites:
#   1. Launch an EC2 instance:
#      - AMI: Ubuntu 22.04 LTS
#      - Instance type: t3.medium (2 vCPU, 4GB RAM) for 20+ users
#      - Storage: 20GB GP3
#      - Security group: allow TCP 8501 from 0.0.0.0/0
#   2. SSH into it: ssh -i your-key.pem ubuntu@<public-ip>
#   3. Run this script: bash aws-ec2.sh
#
# After running, access: http://<public-ip>:8501
# ─────────────────────────────────────────────────────────────────

set -e

echo "=== Installing Docker ==="
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

echo "=== Cloning Repository ==="
git clone https://github.com/Nshg98/Cascade_1.git
cd Cascade_1

echo "=== Building and Starting ==="
sudo docker compose up -d --build

echo ""
echo "=== DEPLOYED ==="
echo "Access at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<your-public-ip>'):8501"
echo ""
echo "Useful commands:"
echo "  sudo docker compose logs -f      # view logs"
echo "  sudo docker compose restart      # restart"
echo "  sudo docker compose down          # stop"
echo "  sudo docker compose up -d --build # rebuild after code changes"

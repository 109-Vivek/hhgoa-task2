# GCP Compute Engine VM Deployment & GitHub Actions CI/CD Guide

This guide walks you through deploying the **Voice-Enabled Indic RAG** system directly on a **Google Cloud Compute Engine (VM)** using your **GCP $300 free credits**, and setting up **Continuous Deployment via GitHub Actions (SSH)**.

---

## Architecture Overview

```
[ Git Push to 'main' ]
         │
         ▼
[ GitHub Actions ] ──(SSH)──► [ GCP Compute Engine VM (Ubuntu) ]
                                    ├── Git Pull & React Build
                                    ├── FastAPI (Port 8000)
                                    ├── Nginx Reverse Proxy (Port 80)
                                    └── Local Index Files (/data/indices)
                                              │
                                              ▼
                             [ Public URL: http://<VM_EXTERNAL_IP> ]
```

---

## Step 1: Create a VM Instance on GCP

1. Go to **[GCP Console > Compute Engine > VM Instances](https://console.cloud.google.com/compute/instances)**.
2. Click **Create Instance**:
   - **Name**: `hhgoa-rag-vm`
   - **Region**: `asia-south1` (Mumbai) or `us-central1`
   - **Machine Type**: `e2-standard-2` (2 vCPUs, 8 GB RAM) or `e2-standard-4` (4 vCPUs, 16 GB RAM for faster indexing) — *Fully covered by $300 credits*.
   - **Boot Disk**: Ubuntu 22.04 LTS or Ubuntu 24.04 LTS (x86/64), **50 GB Balanced Persistent Disk**.
   - **Firewall**: Check both:
     - ✅ **Allow HTTP traffic** (Port 80)
     - ✅ **Allow HTTPS traffic** (Port 443)
3. Click **Create**.
4. Note your **External IP** (e.g., `34.93.xxx.xxx`).

---

## Step 2: SSH into the VM & Run 1-Click Setup

Click the **SSH** button in the GCP Console next to your VM instance, or use local terminal SSH.

### 1. Clone your Repository & Configure `.env`
```bash
# Clone repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/hhgoa.git ~/hhgoa
cd ~/hhgoa

# Create and configure .env
cp .env.example .env
nano .env
# Fill in your SARVAM_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, etc.
```

### 2. Run the 1-Click Provisioning Script
```bash
chmod +x scripts/setup_vm.sh
./scripts/setup_vm.sh
```

*(This automatically installs Python, Node, Nginx, builds the React frontend, and creates a systemd background service `hhgoa.service` connected to Nginx on Port 80).*

---

## Step 3: Run Indexing on the VM

You can run the indexing script in a persistent `tmux` session on the VM so it keeps running even if you close your terminal:

```bash
cd ~/hhgoa
source .venv/bin/activate

# Start a tmux session
tmux new -s indexing

# Run indexing for your desired passage count (e.g., 500 passages per language)
python -m src.indexer --languages gu hi te --limit 500 --strategy metadata_augmented

# (To detach from tmux, press Ctrl+B, then D)
# (To re-attach later, run: tmux attach -t indexing)
```

Once indexing finishes, reload the FastAPI service to load the new indices:
```bash
sudo systemctl restart hhgoa
```

Your app is now live and accessible at: **`http://<YOUR_VM_EXTERNAL_IP>`**!

---

## Step 4: Setup Continuous Deployment with GitHub Actions

Whenever you push code changes to `main`, GitHub Actions will automatically SSH into the VM, pull the latest code, rebuild the React frontend, and restart the backend.

### 1. Generate an SSH Key Pair for GitHub Actions
On your local machine or in the VM terminal, generate a dedicated deploy key:
```bash
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_deploy -N ""
```

Add the public key to the VM's authorized keys:
```bash
cat ~/.ssh/github_deploy.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 2. Allow Passwordless Systemctl Restart for the Deployer User
To allow GitHub Actions to restart the service without being prompted for a sudo password:
```bash
sudo bash -c "echo '$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart hhgoa, /usr/bin/systemctl is-active hhgoa' > /etc/sudoers.d/hhgoa-deploy"
```

### 3. Add GitHub Repository Secrets
In your GitHub repository:
Go to **Settings** ➔ **Secrets and variables** ➔ **Actions** ➔ **New repository secret**

Add these 3 secrets:

| Secret Name | Description / Value |
| :--- | :--- |
| `VM_IP` | The External IP of your GCP VM (e.g. `34.93.123.45`) |
| `VM_USER` | Your SSH username on the VM (e.g. `vivekkumar` or `ubuntu`) |
| `VM_SSH_KEY` | The entire private key content (`cat ~/.ssh/github_deploy`) |

---

## Useful VM Management Commands

```bash
# Check FastAPI service logs
sudo journalctl -u hhgoa -f

# Restart / Status of the app
sudo systemctl restart hhgoa
sudo systemctl status hhgoa

# Check Nginx status / logs
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

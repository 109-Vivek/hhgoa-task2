#!/usr/bin/env bash
# ==============================================================================
# 1-Click VM Setup Script for HH Goa Indic Voice RAG
# Target OS: Ubuntu 22.04 / 24.04 LTS (GCP Compute Engine)
# ==============================================================================

set -e

echo "=========================================="
echo "🚀 Starting HH Goa Indic Voice RAG VM Setup"
echo "=========================================="

# 1. Update System & Install Core Packages
echo "📦 Installing system dependencies (Python, Node.js, Nginx, C/C++ build tools)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    nginx \
    tmux \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg

# 2. Install Node.js 20.x LTS
if ! command -v node &> /dev/null; then
    echo "📦 Installing Node.js 20 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# 3. Setup Project Directory & Python Virtual Environment
APP_DIR="$HOME/hhgoa"
echo "📁 Setting up project in $APP_DIR..."

cd "$APP_DIR"

if [ ! -d ".venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Build Frontend
echo "⚛️ Building React frontend..."
cd "$APP_DIR/frontend"
npm install
npm run build
cd "$APP_DIR"

# 5. Create Data & Indices Directory
mkdir -p "$APP_DIR/data/indices"

# 6. Setup Systemd Service for FastAPI Server
echo "⚙️ Configuring Systemd Service (hhgoa.service)..."
sudo tee /etc/systemd/system/hhgoa.service > /dev/null << EOF
[Unit]
Description=HH Goa Indic Voice RAG FastAPI Server
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn src.server:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable hhgoa
sudo systemctl restart hhgoa

# 7. Configure Nginx Reverse Proxy on Port 80
echo "🌐 Configuring Nginx Reverse Proxy on port 80..."
sudo tee /etc/nginx/sites-available/hhgoa > /dev/null << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 50M;

    # Gzip Compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/hhgoa /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "=========================================="
echo "✅ Setup Complete!"
echo "FastAPI + Nginx is now running on Port 80."
echo "Your app is accessible via http://<YOUR_VM_EXTERNAL_IP>"
echo "=========================================="

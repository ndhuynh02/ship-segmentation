How to run:
1. Clone CVAT's github repository: `git clone https://github.com/opencv/cvat.git`
2. Build docker: `docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build`
3. Install Nuclio: 
    - `wget https://github.com/nuclio/nuclio/releases/download/1.11.24/nuctl-1.11.24-linux-amd64`
    - `sudo chmod +x nuctl-1.11.24-linux-amd64`
    - `sudo ln -sf $(pwd)/nuctl-1.11.24-linux-amd64 /usr/local/bin/nuctl`
4. Run this command: `./cvat/deploy_cpu.sh ./cvat/nuclio/`
5. Go to `localhost:8070` for Nuclio and `localhost:8080` for CVAT
6. Enjoy!
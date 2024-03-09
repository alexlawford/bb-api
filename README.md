Choose the ubuntu/cuda template on vast.ai
Edit template and in “docker options” add “-p 8081:8081”
Set up & star the instance
Follow the instructions for ssh’ing into the the instance
Type uname -a to confirm OS details (expecting Ubuntu)

sudo apt update

sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl

curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg

curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list

sudo apt update

sudo apt install caddy

sudo apt install python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools

sudo apt install python3-venv

git config --global credential.helper store

git clone https://github.com/alexlawford/bb-api.git

python3 -m venv env (follow instructions)

source env/bin/activate

cd bb-api/

pip install -r requirements.txt

sudo caddy start

gunicorn app:app
# 0. clean container within same group
echo "=== [RNE Final Project] Run ==="
echo "[RNE Final Project]  Remove Containers ..."
docker compose -p app down --volumes --remove-orphans

# 1. environment setup  
export COMPOSE_BAKE=true
export DISPLAY=localhost:0.0

# 2. export DISPLAY=:0
xhost +local:docker
cd docker

## 3. startup the container
echo "[RNE Final Project] Launching container ..."
docker compose -p app up dev -d
# docker compose -p app up living-room -d
# docker compose -p app up door-random -d
# docker compose -p app up pikachu -d
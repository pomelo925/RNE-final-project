#!/bin/bash

PORT_MAPPING=""
if [ "$1" = "--port" ] && [ -n "$2" ] && [ -n "$3" ]; then
    PORT_MAPPING="-p $2:$3"
    shift 3
fi

ENV_FIX_SCRIPT="$(pwd)/env_fix.sh"
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network compose_my_bridge_network \
        $PORT_MAPPING \
        --runtime=nvidia \
        --env-file .env \
        -v "$(pwd)/src:/workspace/src" \
        -v "$ENV_FIX_SCRIPT:/tmp/env_fix.sh" \
        --entrypoint bash \
        registry.screamtrumpet.csie.ncku.edu.tw/screamlab/ros2_yolo_opencv_image:latest 
        # -c "chmod +x /tmp/env_fix.sh && /tmp/env_fix.sh && exec bash"

elif [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    echo "Detected architecture: amd64 or macOS arm64"
    if [ "$OS" = "Darwin" ]; then
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            -v "$(pwd)/fps_screenshots:/workspaces/fps_screenshots" \
            -v "$ENV_FIX_SCRIPT:/tmp/env_fix.sh" \
            --entrypoint bash \
            registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 
            # -c "chmod +x /tmp/env_fix.sh && /tmp/env_fix.sh && exec bash"
    else
        echo "Trying to run with GPU support..."
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            --gpus all \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            -v "$(pwd)/fps_screenshots:/workspaces/fps_screenshots" \
            -v "$ENV_FIX_SCRIPT:/tmp/env_fix.sh" \
            --entrypoint bash \
            registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 
            # -c "chmod +x /tmp/env_fix.sh && /tmp/env_fix.sh && exec bash"

        if [ $? -ne 0 ]; then
            echo "GPU not supported or failed, falling back to CPU mode..."
            docker run -it --rm \
                --network compose_my_bridge_network \
                $PORT_MAPPING \
                --env-file .env \
                -v "$(pwd)/src:/workspaces/src" \
                -v "$(pwd)/screenshots:/workspaces/screenshots" \
                -v "$(pwd)/fps_screenshots:/workspaces/fps_screenshots" \
                -v "$ENV_FIX_SCRIPT:/tmp/env_fix.sh" \
                --entrypoint bash \
                registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 
                # -c "chmod +x /tmp/env_fix.sh && /tmp/env_fix.sh && exec bash"
        fi
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

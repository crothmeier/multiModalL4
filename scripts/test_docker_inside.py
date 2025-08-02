#!/usr/bin/env python3
import os

import docker

print(f"Docker version: {docker.__version__}")
print(f"DOCKER_HOST: {os.environ.get('DOCKER_HOST', 'not set')}")

# Try different connection methods
try:
    client = docker.from_env()
    print("docker.from_env() worked!")
except Exception as e:
    print(f"docker.from_env() failed: {e}")

try:
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    print("DockerClient with unix://var/run/docker.sock worked!")
except Exception as e:
    print(f"DockerClient failed: {e}")

try:
    client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
    print("DockerClient with unix:///var/run/docker.sock worked!")
    print(f"Docker info: {client.info()['ServerVersion']}")
except Exception as e:
    print(f"DockerClient with triple slash failed: {e}")

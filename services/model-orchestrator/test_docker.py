#!/usr/bin/env python3
import os
import sys

import requests

# Check environment
print("Environment:")
print(f"DOCKER_HOST: {os.environ.get('DOCKER_HOST', 'not set')}")
print(f"Python version: {sys.version}")

# Test direct connection
try:
    # Direct connection to Docker socket
    session = requests.Session()
    session.mount("http+docker://", requests.adapters.HTTPAdapter())

    # Try via unix socket
    import requests_unixsocket

    session = requests_unixsocket.Session()
    response = session.get("http+unix://%2Fvar%2Frun%2Fdocker.sock/version")
    print(f"Direct socket test: {response.status_code}")
except Exception as e:
    print(f"Direct socket error: {e}")

# Test with docker-py
try:
    import docker

    client = docker.from_env()
    version = client.version()
    print(f"Docker version: {version['Version']}")
    print("Docker connection successful!")
except Exception as e:
    print(f"Docker-py error: {e}")

    # Try alternative connection
    try:
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        version = client.version()
        print(f"Alternative connection worked: {version['Version']}")
    except Exception as e2:
        print(f"Alternative connection error: {e2}")

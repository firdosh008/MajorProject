import os
from dotenv import load_dotenv
import subprocess
import time
import signal
import sys

# Load environment variables from .env file
load_dotenv()

def run_command(command):
    return subprocess.Popen(command, shell=True)

# Define cleanup handler for graceful shutdown
def cleanup_handler(signum, frame):
    print("\nShutting down services...")
    for process in backend_processes + client_processes:
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
    print("All processes terminated")
    sys.exit(0)

# Register signal handlers for graceful termination
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)

# Run backend services
backend_services = [
    "python RunMaster.py",
    "python RunDetect.py",
    "python RunTracker.py",
    "python RunCrash.py"
]

# Run client services (just process video files, no streaming)
client_services = [
    "python RunCamera.py"
]

print("Starting backend services...")
backend_processes = [run_command(service) for service in backend_services]

# Wait a bit for backend services to initialize
time.sleep(5)
print("Backend services started")

print("Starting client services...")
client_processes = [run_command(service) for service in client_services]
print("Client services started")

print("Argus system is running. Press Ctrl+C to stop.")

# Wait for all processes to complete or user interruption
try:
    while all(process.poll() is None for process in backend_processes + client_processes):
        time.sleep(1)
except KeyboardInterrupt:
    cleanup_handler(None, None)

# If we get here, some process has exited
print("Some process has exited. Shutting down all services...")
cleanup_handler(None, None)
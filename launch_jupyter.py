import modal
import subprocess
import time
import os

GPU_CONFIG = modal.gpu.A10G(count=1)
MINUTES = 60

volume = modal.Volume.from_name("chameleon-model", create_if_missing=True)

model_store_path = "/vol/models/chameleon"

meta_chameleon_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .run_commands("pip install git+https://github.com/facebookresearch/chameleon.git", force_build=True)
    .pip_install("jupyterlab")
)

JUPYTER_TOKEN = "1234" 

app = modal.App("chameleon-jupyter", image = meta_chameleon_image)

@app.function(concurrency_limit=1, volumes={model_store_path: volume}, timeout= 30000, gpu = GPU_CONFIG)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 30000):
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)

#modal run chameleon_jupyter_server.py




import subprocess

subprocess.run([
    "mlflow",
    "ui",
    "--backend-store-uri", "sqlite:////app/mlflow.db",
    "--host","0.0.0.0",
    "--allowed-hosts",  "localhost,127.0.0.1",
    "--cors-allowed-origins", "http://localhost:5000,http://127.0.0.1:5000",
    "--port","5000",
])


# pip uninstall -y mlflow mlflow-skinny mlflow-tracing

# pip freeze | grep mlflow

# pip install --no-cache-dir mlflow==2.22.1

# mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
# export GIT_PYTHON_REFRESH=quiet
# sudo chown -R $USER:$USER ~/Project/tryMsflowMnist
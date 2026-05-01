Write-Host "=========================================="
Write-Host "  DBN MNIST Project - Setup"
Write-Host "=========================================="

Write-Host "Rebuilding virtual environment..."
python -m venv .venv --clear

Write-Host "Upgrading pip..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing standard dependencies..."
.\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Installing deep-belief-network from GitHub..."
.\.venv\Scripts\pip.exe install git+https://github.com/albertbup/deep-belief-network.git --no-deps

Write-Host "=========================================="
Write-Host "  Setup complete!"
Write-Host "  Activate the environment with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "  Then run the project with:"
Write-Host "    python main.py"
Write-Host "=========================================="

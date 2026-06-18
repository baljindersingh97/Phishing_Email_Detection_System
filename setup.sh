
set -e
GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[+]${NC} $1"; }

cd "$(dirname "$0")"

echo ""
echo "========================================================"
echo "  Phishing Email Detection System — Setup"
echo "  B207 Cyber Security | Gisma University of Applied Sciences"
echo "  Author: Baljinder Singh"
echo "========================================================"
echo ""

# 1. Check Python version
PYTHON=$(command -v python3 || command -v python)
$PYTHON -c "import sys; assert sys.version_info>=(3,8)" 2>/dev/null \
  || { echo "Python 3.8+ is required."; exit 1; }
info "Python OK  ($($PYTHON --version))"

# 2. Virtual environment
[ -d venv ] || $PYTHON -m venv venv
source venv/bin/activate
info "Virtual environment ready"

# 3. Install dependencies
pip install --upgrade pip -q
pip install -r requirements.txt -q
info "Dependencies installed (scikit-learn, pandas, numpy, scipy)"

# 4. Initialise database
info "Initialising SQLite database..."
python -c "from detector import init_db; init_db()"

# 5. Train model
info "Training model (this takes about 30 seconds)..."
python train.py

echo ""
echo "========================================================"
echo "  Setup complete! Try these commands:"
echo ""
echo "  source venv/bin/activate"
echo ""
echo '  python analyse.py --text "URGENT verify your PayPal NOW!"'
echo "  python analyse.py --file path/to/email.txt"
echo "  python analyse.py --interactive"
echo "  python analyse.py --history"
echo ""
echo "  Kaggle dataset (for best accuracy):"
echo "  https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset"
echo "  Place CSV at: data/phishing_email.csv then re-run: python train.py"
echo "========================================================"
echo ""

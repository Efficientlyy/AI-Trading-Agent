#\!/bin/bash
# Simple script to start the modern dashboard with the correct template paths

export FLASK_APP=run_modern_dashboard.py
export FLASK_DEBUG=1
export FLASK_ENV=development
export FLASK_TEMPLATE_FOLDER="$(pwd)/templates"
export FLASK_STATIC_FOLDER="$(pwd)/static"

echo "Starting dashboard with template directory: $FLASK_TEMPLATE_FOLDER"
echo "Starting dashboard with static directory: $FLASK_STATIC_FOLDER"

python run_modern_dashboard.py --debug


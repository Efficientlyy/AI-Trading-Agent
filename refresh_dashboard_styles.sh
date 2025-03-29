#!/bin/bash

echo "Refreshing dashboard styles..."

# Get current timestamp
TIMESTAMP=$(date +%s)

# Update API keys panel CSS with timestamp
cp static/css/api_keys_panel.css static/css/api_keys_panel_${TIMESTAMP}.css

# Create a launcher script
cat > run_fixed_dashboard_temp.sh << EOF
#!/bin/bash

# Start the dashboard
echo "Running dashboard with refreshed styles..."
export FLASK_APP=fixed_dashboard.py
export FLASK_DEBUG=1
export FLASK_ENV=development
python fixed_dashboard.py --debug
EOF

chmod +x run_fixed_dashboard_temp.sh

# Update HTML template to use the new CSS file
sed -i "s|api_keys_panel.css|api_keys_panel_${TIMESTAMP}.css|g" templates/dropdown_fix.html

echo "Dashboard styles refreshed. Run ./run_fixed_dashboard_temp.sh to start the dashboard."
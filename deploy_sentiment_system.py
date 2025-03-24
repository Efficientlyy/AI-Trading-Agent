#!/usr/bin/env python
"""
Deployment script for the sentiment analysis system.

This script automates the deployment process for the enhanced sentiment analysis system,
including LLM integration, consensus system, real-time event detection, and dashboard.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Banner and styling constants
BANNER = """
╭───────────────────────────────────────────────────────╮
│                                                       │
│       SENTIMENT ANALYSIS SYSTEM DEPLOYMENT            │
│                                                       │
╰───────────────────────────────────────────────────────╯
"""

HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"


class SentimentSystemDeployer:
    """Automated deployment for the sentiment analysis system."""
    
    def __init__(self, config_path: str, environment: str, skip_tests: bool = False):
        """Initialize the deployer.
        
        Args:
            config_path: Path to the configuration file
            environment: Deployment environment (dev, staging, prod)
            skip_tests: Whether to skip running tests
        """
        self.config_path = config_path
        self.environment = environment
        self.skip_tests = skip_tests
        
        # Load config
        self.config = self._load_config()
        
        # Create deployment directory if it doesn't exist
        self.deployment_dir = f"deployments/{environment}"
        os.makedirs(self.deployment_dir, exist_ok=True)
        
        # Initialize logs
        self.deployment_log = []
        self.log_path = f"{self.deployment_dir}/deployment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        print(f"{BLUE}Loading configuration from: {self.config_path}{ENDC}")
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Validate config has required sections
            required_sections = ['sentiment_analysis', 'llm', 'early_detection']
            missing = [section for section in required_sections if section not in config]
            
            if missing:
                print(f"{RED}Missing required configuration sections: {', '.join(missing)}{ENDC}")
                sys.exit(1)
            
            return config
        except Exception as e:
            print(f"{RED}Error loading configuration: {str(e)}{ENDC}")
            sys.exit(1)
    
    def _log_step(self, step: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a deployment step.
        
        Args:
            step: Name of the deployment step
            status: Status of the step (success, error, warning)
            details: Optional details about the step
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "details": details or {}
        }
        
        self.deployment_log.append(log_entry)
        
        # Write logs after each step to prevent loss in case of failure
        with open(self.log_path, 'w') as f:
            json.dump(self.deployment_log, f, indent=2)
        
        # Print step status
        status_color = GREEN if status == "success" else YELLOW if status == "warning" else RED
        print(f"{status_color}{step}: {status.upper()}{ENDC}")
        
        if details and status != "success":
            print(f"  Details: {json.dumps(details, indent=2)}")
    
    async def validate_api_keys(self) -> bool:
        """Validate API keys are available.
        
        Returns:
            True if all required API keys are available
        """
        print(f"{BLUE}Validating API keys...{ENDC}")
        
        # Required API keys
        required_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "TWITTER_API_KEY": os.getenv("TWITTER_API_KEY"),
            "CRYPTOCOMPARE_API_KEY": os.getenv("CRYPTOCOMPARE_API_KEY"),
            "NEWS_API_KEY": os.getenv("NEWS_API_KEY")
        }
        
        # Check for optional keys
        optional_keys = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY")
        }
        
        # Validate required keys
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            self._log_step("validate_api_keys", "error", {"missing_keys": missing_keys})
            print(f"{RED}Missing required API keys: {', '.join(missing_keys)}{ENDC}")
            print(f"{YELLOW}Please set these environment variables before deployment.{ENDC}")
            return False
        
        # Warn about optional keys
        missing_optional = [key for key, value in optional_keys.items() if not value]
        
        if missing_optional:
            self._log_step("validate_api_keys", "warning", {"missing_optional_keys": missing_optional})
            print(f"{YELLOW}Missing optional API keys: {', '.join(missing_optional)}{ENDC}")
            print(f"{YELLOW}These keys are not required, but provide additional functionality.{ENDC}")
        else:
            self._log_step("validate_api_keys", "success")
        
        return True
    
    async def run_tests(self) -> bool:
        """Run system tests.
        
        Returns:
            True if all tests pass
        """
        if self.skip_tests:
            print(f"{YELLOW}Skipping tests as requested.{ENDC}")
            self._log_step("run_tests", "skipped")
            return True
        
        print(f"{BLUE}Running system tests...{ENDC}")
        
        try:
            # Run the main sentiment system test
            print(f"{BLUE}Running sentiment system tests...{ENDC}")
            process = subprocess.run(
                [sys.executable, "test_sentiment_system.py"],
                capture_output=True,
                text=True,
                check=False
            )
            
            tests_passed = process.returncode == 0
            
            if tests_passed:
                self._log_step("run_tests", "success")
                print(f"{GREEN}All tests passed!{ENDC}")
            else:
                self._log_step("run_tests", "error", {
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr
                })
                print(f"{RED}Tests failed!{ENDC}")
                print(f"{RED}Error: {process.stderr}{ENDC}")
                
                # If in production environment, exit on test failure
                if self.environment == "prod":
                    print(f"{RED}Aborting deployment to production due to test failures.{ENDC}")
                    return False
                
                # For other environments, warn but continue
                print(f"{YELLOW}Continuing deployment despite test failures (non-production environment).{ENDC}")
            
            return True
            
        except Exception as e:
            self._log_step("run_tests", "error", {"error": str(e)})
            print(f"{RED}Error running tests: {str(e)}{ENDC}")
            
            # If in production environment, exit on test error
            if self.environment == "prod":
                return False
            
            # For other environments, warn but continue
            print(f"{YELLOW}Continuing deployment despite test errors (non-production environment).{ENDC}")
            return True
    
    async def update_configurations(self) -> bool:
        """Update system configurations.
        
        Returns:
            True if configurations were updated successfully
        """
        print(f"{BLUE}Updating configurations...{ENDC}")
        
        config_files = {
            "sentiment_analysis": "config/sentiment_analysis.yaml",
            "early_detection": "config/early_detection.yaml",
            "dashboard": "config/dashboard.yaml"
        }
        
        success = True
        
        # Create backups
        for config_name, config_path in config_files.items():
            if os.path.exists(config_path):
                backup_path = f"{config_path}.bak"
                shutil.copy2(config_path, backup_path)
                print(f"{BLUE}Backed up {config_path} to {backup_path}{ENDC}")
        
        try:
            # Update sentiment_analysis.yaml
            sentiment_config_path = config_files["sentiment_analysis"]
            with open(sentiment_config_path, 'r') as f:
                sentiment_config = yaml.safe_load(f)
            
            # Apply LLM and consensus system settings
            if "llm" in self.config:
                sentiment_config["llm"] = self.config["llm"]
            
            if "consensus_system" in self.config.get("sentiment_analysis", {}):
                sentiment_config["consensus_system"] = self.config["sentiment_analysis"]["consensus_system"]
            
            # Write updated config
            with open(sentiment_config_path, 'w') as f:
                yaml.dump(sentiment_config, f, default_flow_style=False, sort_keys=False)
            
            # Update early_detection.yaml
            detection_config_path = config_files["early_detection"]
            with open(detection_config_path, 'r') as f:
                detection_config = yaml.safe_load(f)
            
            # Apply early detection settings
            if "early_detection" in self.config:
                detection_config.update(self.config["early_detection"])
            
            # Write updated config
            with open(detection_config_path, 'w') as f:
                yaml.dump(detection_config, f, default_flow_style=False, sort_keys=False)
            
            # Update dashboard.yaml if it exists
            dashboard_config_path = config_files["dashboard"]
            if os.path.exists(dashboard_config_path):
                with open(dashboard_config_path, 'r') as f:
                    dashboard_config = yaml.safe_load(f)
                
                # Apply dashboard settings
                if "dashboard" in self.config:
                    dashboard_config.update(self.config["dashboard"])
                
                # Write updated config
                with open(dashboard_config_path, 'w') as f:
                    yaml.dump(dashboard_config, f, default_flow_style=False, sort_keys=False)
            
            self._log_step("update_configurations", "success")
            
        except Exception as e:
            self._log_step("update_configurations", "error", {"error": str(e)})
            print(f"{RED}Error updating configurations: {str(e)}{ENDC}")
            
            # Restore backups
            for config_name, config_path in config_files.items():
                backup_path = f"{config_path}.bak"
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, config_path)
                    print(f"{YELLOW}Restored {config_path} from backup{ENDC}")
            
            success = False
        
        return success
    
    async def verify_deployment(self) -> bool:
        """Verify deployment by running a simple check.
        
        Returns:
            True if verification is successful
        """
        print(f"{BLUE}Verifying deployment...{ENDC}")
        
        try:
            # Import key components to verify they can be loaded
            from src.analysis_agents.sentiment.llm_service import LLMService
            from src.analysis_agents.sentiment.llm_sentiment_agent import LLMSentimentAgent
            from src.analysis_agents.sentiment.consensus_system import ConsensusSystem
            from src.analysis_agents.early_detection.realtime_detector import RealtimeEventDetector
            from src.dashboard.llm_event_dashboard import LLMEventDashboard
            
            # Create an LLM service instance and test initialization
            llm_service = LLMService()
            await llm_service.initialize()
            
            # Close service
            await llm_service.close()
            
            self._log_step("verify_deployment", "success")
            return True
            
        except Exception as e:
            self._log_step("verify_deployment", "error", {"error": str(e)})
            print(f"{RED}Error verifying deployment: {str(e)}{ENDC}")
            return False
    
    async def generate_deployment_report(self) -> None:
        """Generate a deployment report."""
        print(f"{BLUE}Generating deployment report...{ENDC}")
        
        report = {
            "deployment_id": f"deploy-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "components": [
                "llm_service",
                "llm_sentiment_agent",
                "consensus_system",
                "realtime_detector",
                "llm_event_dashboard"
            ],
            "configuration": {
                key: value for key, value in self.config.items() 
                if key not in ["api_keys"]  # Don't include sensitive data
            },
            "log": self.deployment_log
        }
        
        # Generate HTML report
        html_report_path = f"{self.deployment_dir}/deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(html_report_path, 'w') as f:
            # HTML header
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sentiment Analysis System Deployment Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333; }
                    .header { text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
                    .success { color: green; }
                    .error { color: red; }
                    .warning { color: orange; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                    .log-entry { margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
                    pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
                </style>
            </head>
            <body>
            """)
            
            # Report header
            f.write(f"""
            <div class="header">
                <h1>Sentiment Analysis System Deployment Report</h1>
                <p>Environment: {self.environment.upper()}</p>
                <p>Deployment ID: {report['deployment_id']}</p>
                <p>Timestamp: {report['timestamp']}</p>
            </div>
            """)
            
            # Components
            f.write("""
            <div class="section">
                <h2>Deployed Components</h2>
                <ul>
            """)
            
            for component in report['components']:
                f.write(f"<li>{component}</li>")
            
            f.write("""
                </ul>
            </div>
            """)
            
            # Configuration
            f.write("""
            <div class="section">
                <h2>Configuration</h2>
                <pre>""")
            
            # Format the config as pretty JSON
            f.write(json.dumps(report['configuration'], indent=2))
            
            f.write("""</pre>
            </div>
            """)
            
            # Deployment Log
            f.write("""
            <div class="section">
                <h2>Deployment Log</h2>
            """)
            
            for entry in report['log']:
                status_class = "success" if entry['status'] == "success" else "warning" if entry['status'] == "warning" else "error"
                
                f.write(f"""
                <div class="log-entry">
                    <h3 class="{status_class}">{entry['step']} - {entry['status'].upper()}</h3>
                    <p>Timestamp: {entry['timestamp']}</p>
                """)
                
                if entry['details']:
                    f.write(f"""
                    <div>
                        <h4>Details</h4>
                        <pre>{json.dumps(entry['details'], indent=2)}</pre>
                    </div>
                    """)
                
                f.write("</div>")
            
            f.write("""
            </div>
            """)
            
            # HTML footer
            f.write("""
            </body>
            </html>
            """)
        
        print(f"{GREEN}Deployment report generated: {html_report_path}{ENDC}")
        self._log_step("generate_deployment_report", "success", {"report_path": html_report_path})
    
    async def deploy(self) -> bool:
        """Run the full deployment process.
        
        Returns:
            True if deployment was successful
        """
        print(BANNER)
        print(f"{BOLD}Deploying Sentiment Analysis System to {self.environment.upper()} environment{ENDC}")
        print(f"{BOLD}Configuration: {self.config_path}{ENDC}")
        print()
        
        # Run deployment steps
        api_keys_valid = await self.validate_api_keys()
        if not api_keys_valid:
            return False
        
        tests_passed = await self.run_tests()
        if not tests_passed and self.environment == "prod":
            return False
        
        configs_updated = await self.update_configurations()
        if not configs_updated:
            return False
        
        verified = await self.verify_deployment()
        
        # Generate deployment report
        await self.generate_deployment_report()
        
        if not verified:
            print(f"{RED}Deployment verification failed!{ENDC}")
            return False
        
        print(f"{GREEN}Deployment completed successfully!{ENDC}")
        return True


def main():
    """Main function to run the deployment script."""
    parser = argparse.ArgumentParser(description="Deploy the Sentiment Analysis System")
    parser.add_argument("--config", type=str, required=True, help="Path to the deployment configuration file")
    parser.add_argument("--environment", type=str, default="dev", choices=["dev", "staging", "prod"], 
                        help="Deployment environment (dev, staging, prod)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running system tests")
    
    args = parser.parse_args()
    
    # Create and run deployer
    deployer = SentimentSystemDeployer(args.config, args.environment, args.skip_tests)
    success = asyncio.run(deployer.deploy())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
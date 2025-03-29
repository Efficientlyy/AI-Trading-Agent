#!/usr/bin/env python
"""
Verification script for sentiment analysis system deployment.

This script checks if all components are properly installed and configured.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Banner and styling constants
BANNER = """
╭───────────────────────────────────────────────────────╮
│                                                       │
│    SENTIMENT ANALYSIS SYSTEM DEPLOYMENT VERIFICATION  │
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


class SentimentVerifier:
    """Verification for the sentiment analysis system deployment."""
    
    def __init__(self, environment: str, detailed: bool = False):
        """Initialize the verifier.
        
        Args:
            environment: Deployment environment (dev, staging, prod)
            detailed: Whether to run detailed verification tests
        """
        self.environment = environment
        self.detailed = detailed
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "components": {},
            "verification": {}
        }
    
    async def verify_imports(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify all required modules can be imported.
        
        Returns:
            (success, results)
        """
        print(f"{BLUE}Verifying imports...{ENDC}")
        
        required_modules = [
            "src.analysis_agents.sentiment.llm_service",
            "src.analysis_agents.sentiment.llm_sentiment_agent",
            "src.analysis_agents.sentiment.consensus_system",
            "src.analysis_agents.early_detection.realtime_detector",
            "src.analysis_agents.early_detection.sentiment_integration",
            "src.dashboard.llm_event_dashboard",
            "src.analysis_agents.sentiment.performance_tracker"
        ]
        
        import_results = {}
        all_imports_ok = True
        
        for module in required_modules:
            try:
                # Try to import the module
                __import__(module)
                import_results[module] = {"status": "success"}
                print(f"{GREEN}✓ {module}{ENDC}")
            except ImportError as e:
                import_results[module] = {"status": "error", "error": str(e)}
                print(f"{RED}✗ {module}: {str(e)}{ENDC}")
                all_imports_ok = False
            except Exception as e:
                import_results[module] = {"status": "error", "error": str(e)}
                print(f"{RED}✗ {module}: Unexpected error: {str(e)}{ENDC}")
                all_imports_ok = False
        
        return all_imports_ok, import_results
    
    async def verify_configurations(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify all required configurations are present.
        
        Returns:
            (success, results)
        """
        print(f"{BLUE}Verifying configurations...{ENDC}")
        
        from src.common.config import config
        
        required_configs = [
            "sentiment_analysis.enabled",
            "sentiment_analysis.consensus_system.enabled",
            "early_detection.enabled",
            "early_detection.realtime_detector.enabled",
            "llm.primary_model"
        ]
        
        config_results = {}
        all_configs_ok = True
        
        for config_path in required_configs:
            try:
                # Check if the config key exists
                value = config.get(config_path)
                
                if value is None:
                    config_results[config_path] = {"status": "error", "error": "Missing configuration"}
                    print(f"{RED}✗ {config_path}: Missing configuration{ENDC}")
                    all_configs_ok = False
                else:
                    config_results[config_path] = {"status": "success", "value": str(value)}
                    print(f"{GREEN}✓ {config_path} = {value}{ENDC}")
            except Exception as e:
                config_results[config_path] = {"status": "error", "error": str(e)}
                print(f"{RED}✗ {config_path}: {str(e)}{ENDC}")
                all_configs_ok = False
        
        return all_configs_ok, config_results
    
    async def verify_api_keys(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify required API keys are available.
        
        Returns:
            (success, results)
        """
        print(f"{BLUE}Verifying API keys...{ENDC}")
        
        required_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "TWITTER_API_KEY": os.getenv("TWITTER_API_KEY"),
            "CRYPTOCOMPARE_API_KEY": os.getenv("CRYPTOCOMPARE_API_KEY"),
            "NEWS_API_KEY": os.getenv("NEWS_API_KEY")
        }
        
        optional_keys = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY")
        }
        
        api_results = {}
        all_apis_ok = True
        
        # Check required keys
        for key_name, key_value in required_keys.items():
            if not key_value:
                api_results[key_name] = {"status": "error", "present": False}
                print(f"{RED}✗ {key_name}: Missing{ENDC}")
                all_apis_ok = False
            else:
                # Mask the key value for security
                masked_value = key_value[:4] + "..." + key_value[-4:] if len(key_value) > 8 else "***"
                api_results[key_name] = {"status": "success", "present": True, "value": masked_value}
                print(f"{GREEN}✓ {key_name}: {masked_value}{ENDC}")
        
        # Check optional keys
        for key_name, key_value in optional_keys.items():
            if not key_value:
                api_results[key_name] = {"status": "warning", "present": False}
                print(f"{YELLOW}⚠ {key_name}: Missing (optional){ENDC}")
            else:
                # Mask the key value for security
                masked_value = key_value[:4] + "..." + key_value[-4:] if len(key_value) > 8 else "***"
                api_results[key_name] = {"status": "success", "present": True, "value": masked_value}
                print(f"{GREEN}✓ {key_name}: {masked_value}{ENDC}")
        
        return all_apis_ok, api_results
    
    async def verify_component_initialization(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify that key components can be initialized.
        
        Returns:
            (success, results)
        """
        print(f"{BLUE}Verifying component initialization...{ENDC}")
        
        init_results = {}
        all_inits_ok = True
        
        # Import required components
        try:
            from src.analysis_agents.sentiment.llm_service import LLMService
            from src.analysis_agents.sentiment.performance_tracker import performance_tracker
            from src.analysis_agents.sentiment.consensus_system import ConsensusSystem
            from src.analysis_agents.early_detection.realtime_detector import RealtimeEventDetector
            
            # Try to initialize LLM service (this is critical)
            print(f"{BLUE}Initializing LLM service...{ENDC}")
            llm_service = LLMService()
            await llm_service.initialize()
            init_results["llm_service"] = {"status": "success"}
            print(f"{GREEN}✓ LLM service initialized{ENDC}")
            
            # Close the service after testing
            await llm_service.close()
            
            # Try to initialize performance tracker
            print(f"{BLUE}Initializing performance tracker...{ENDC}")
            await performance_tracker.initialize()
            init_results["performance_tracker"] = {"status": "success"}
            print(f"{GREEN}✓ Performance tracker initialized{ENDC}")
            
            # Try to initialize consensus system
            print(f"{BLUE}Initializing consensus system...{ENDC}")
            consensus_system = ConsensusSystem()
            await consensus_system.initialize()
            init_results["consensus_system"] = {"status": "success"}
            print(f"{GREEN}✓ Consensus system initialized{ENDC}")
            
            # Try to initialize realtime event detector
            print(f"{BLUE}Initializing realtime event detector...{ENDC}")
            realtime_detector = RealtimeEventDetector()
            await realtime_detector.initialize()
            init_results["realtime_detector"] = {"status": "success"}
            print(f"{GREEN}✓ Realtime event detector initialized{ENDC}")
            
        except Exception as e:
            component_name = e.__traceback__.tb_frame.f_globals['__name__']
            init_results[component_name] = {"status": "error", "error": str(e)}
            print(f"{RED}✗ {component_name}: {str(e)}{ENDC}")
            all_inits_ok = False
        
        return all_inits_ok, init_results
    
    async def verify_detailed(self) -> Tuple[bool, Dict[str, Any]]:
        """Run detailed verification tests.
        
        Returns:
            (success, results)
        """
        if not self.detailed:
            return True, {"status": "skipped"}
        
        print(f"{BLUE}Running detailed verification tests...{ENDC}")
        
        detailed_results = {}
        all_tests_ok = True
        
        try:
            # Import required components
            from src.analysis_agents.sentiment.llm_service import LLMService
            
            # Initialize LLM service
            llm_service = LLMService()
            await llm_service.initialize()
            
            # Test sentiment analysis
            print(f"{BLUE}Testing sentiment analysis...{ENDC}")
            sentiment_result = await llm_service.analyze_sentiment(
                "Bitcoin adoption is growing rapidly among institutional investors."
            )
            
            if "sentiment_value" in sentiment_result:
                detailed_results["sentiment_analysis"] = {
                    "status": "success", 
                    "result": {
                        "value": sentiment_result["sentiment_value"],
                        "direction": sentiment_result["direction"],
                        "confidence": sentiment_result["confidence"]
                    }
                }
                print(f"{GREEN}✓ Sentiment analysis: {sentiment_result['direction']} ({sentiment_result['sentiment_value']:.2f}){ENDC}")
            else:
                detailed_results["sentiment_analysis"] = {"status": "error", "error": "Invalid result format"}
                print(f"{RED}✗ Sentiment analysis: Invalid result format{ENDC}")
                all_tests_ok = False
            
            # Test event detection
            print(f"{BLUE}Testing event detection...{ENDC}")
            event_result = await llm_service.detect_market_event(
                "Breaking: SEC approves first spot Bitcoin ETF applications."
            )
            
            if "is_market_event" in event_result:
                detailed_results["event_detection"] = {
                    "status": "success", 
                    "result": {
                        "is_event": event_result["is_market_event"],
                        "event_type": event_result["event_type"],
                        "severity": event_result["severity"],
                        "propagation_speed": event_result["propagation_speed"]
                    }
                }
                print(f"{GREEN}✓ Event detection: {'Event detected' if event_result['is_market_event'] else 'No event'}{ENDC}")
            else:
                detailed_results["event_detection"] = {"status": "error", "error": "Invalid result format"}
                print(f"{RED}✗ Event detection: Invalid result format{ENDC}")
                all_tests_ok = False
            
            # Close the service after testing
            await llm_service.close()
            
        except Exception as e:
            test_name = e.__traceback__.tb_frame.f_globals['__name__']
            detailed_results[test_name] = {"status": "error", "error": str(e)}
            print(f"{RED}✗ Detailed test error: {str(e)}{ENDC}")
            all_tests_ok = False
        
        return all_tests_ok, detailed_results
    
    async def run_verification(self) -> bool:
        """Run all verification steps.
        
        Returns:
            True if all verification passed
        """
        print(BANNER)
        print(f"{BOLD}Verifying Sentiment Analysis System in {self.environment.upper()} environment{ENDC}")
        print()
        
        # Run verification steps
        imports_ok, import_results = await self.verify_imports()
        self.results["components"]["imports"] = import_results
        
        configs_ok, config_results = await self.verify_configurations()
        self.results["components"]["configurations"] = config_results
        
        apis_ok, api_results = await self.verify_api_keys()
        self.results["components"]["api_keys"] = api_results
        
        # Only continue with initialization if imports are OK
        if imports_ok:
            inits_ok, init_results = await self.verify_component_initialization()
            self.results["components"]["initialization"] = init_results
            
            # Run detailed tests if requested and initialization passed
            if inits_ok and self.detailed:
                detailed_ok, detailed_results = await self.verify_detailed()
                self.results["verification"]["detailed"] = detailed_results
            else:
                detailed_ok = True
        else:
            print(f"{RED}Skipping component initialization due to import errors{ENDC}")
            inits_ok = False
            detailed_ok = False
        
        # Overall verification status
        verification_passed = imports_ok and configs_ok and apis_ok and inits_ok and detailed_ok
        
        self.results["overall_status"] = "passed" if verification_passed else "failed"
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary(verification_passed)
        
        return verification_passed
    
    def _save_results(self) -> None:
        """Save verification results to file."""
        # Create verification directory
        os.makedirs("verification", exist_ok=True)
        
        # Save JSON results
        result_path = f"verification/sentiment_verification_{self.environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nVerification results saved to: {result_path}")
    
    def _print_summary(self, verification_passed: bool) -> None:
        """Print a summary of verification results.
        
        Args:
            verification_passed: Whether all verification passed
        """
        print("\n" + "=" * 80)
        if verification_passed:
            print(f"{GREEN}{BOLD}VERIFICATION PASSED!{ENDC}")
            print(f"{GREEN}The sentiment analysis system is properly deployed and configured.{ENDC}")
        else:
            print(f"{RED}{BOLD}VERIFICATION FAILED!{ENDC}")
            print(f"{RED}The sentiment analysis system has issues that need to be addressed.{ENDC}")
        
        print("=" * 80)
        print("\nComponent Status:")
        
        # Imports
        import_status = all(result["status"] == "success" for result in self.results["components"].get("imports", {}).values())
        status_symbol = f"{GREEN}✓{ENDC}" if import_status else f"{RED}✗{ENDC}"
        print(f"  {status_symbol} Imports")
        
        # Configurations
        config_status = all(result["status"] == "success" for result in self.results["components"].get("configurations", {}).values())
        status_symbol = f"{GREEN}✓{ENDC}" if config_status else f"{RED}✗{ENDC}"
        print(f"  {status_symbol} Configurations")
        
        # API Keys
        api_status = all(result["status"] == "success" for key, result in self.results["components"].get("api_keys", {}).items() 
                        if result["status"] != "warning")
        status_symbol = f"{GREEN}✓{ENDC}" if api_status else f"{RED}✗{ENDC}"
        print(f"  {status_symbol} API Keys")
        
        # Initialization
        init_status = all(result["status"] == "success" for result in self.results["components"].get("initialization", {}).values())
        if "initialization" in self.results["components"]:
            status_symbol = f"{GREEN}✓{ENDC}" if init_status else f"{RED}✗{ENDC}"
            print(f"  {status_symbol} Component Initialization")
        
        # Detailed Verification
        if self.detailed and "detailed" in self.results.get("verification", {}):
            detailed_status = all(result["status"] == "success" for result in self.results["verification"]["detailed"].values() 
                                if result["status"] != "skipped")
            status_symbol = f"{GREEN}✓{ENDC}" if detailed_status else f"{RED}✗{ENDC}"
            print(f"  {status_symbol} Detailed Verification")


def main():
    """Main function to run the verification script."""
    parser = argparse.ArgumentParser(description="Verify Sentiment Analysis System Deployment")
    parser.add_argument("--environment", type=str, default="dev", choices=["dev", "staging", "prod"], 
                        help="Deployment environment (dev, staging, prod)")
    parser.add_argument("--detailed", action="store_true", help="Run detailed verification with LLM API calls")
    
    args = parser.parse_args()
    
    # Create and run verifier
    verifier = SentimentVerifier(args.environment, args.detailed)
    verification_passed = asyncio.run(verifier.run_verification())
    
    # Exit with appropriate status code
    sys.exit(0 if verification_passed else 1)


if __name__ == "__main__":
    main()
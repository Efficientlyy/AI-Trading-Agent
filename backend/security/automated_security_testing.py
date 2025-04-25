"""
Automated Security Testing

This module provides integration with security testing tools like OWASP ZAP
for automated security scanning and vulnerability detection in CI/CD pipelines.
"""

import logging
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import requests
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger("security.testing")


class VulnerabilitySeverity(str):
    """Vulnerability severity levels"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SecurityVulnerability(BaseModel):
    """Model for security vulnerabilities"""
    id: str
    name: str
    description: str
    severity: str
    evidence: Optional[str] = None
    url: Optional[str] = None
    param: Optional[str] = None
    attack: Optional[str] = None
    solution: Optional[str] = None
    cwe_id: Optional[int] = None
    wasc_id: Optional[int] = None
    source: str = "unknown"
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.dict(exclude_none=True)


class SecurityScanResult(BaseModel):
    """Model for security scan results"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scan_type: str
    target: str
    total_vulnerabilities: int = 0
    vulnerabilities_by_severity: Dict[str, int] = Field(default_factory=dict)
    vulnerabilities: List[SecurityVulnerability] = Field(default_factory=list)
    scan_id: str
    duration_seconds: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ZAPScanner:
    """
    OWASP ZAP scanner integration
    
    This class provides methods to start ZAP, run scans, and process results.
    """
    
    def __init__(self, 
                zap_path: Optional[str] = None, 
                api_key: Optional[str] = None,
                host: str = "localhost",
                port: int = 8080):
        """
        Initialize ZAP scanner
        
        Args:
            zap_path: Path to ZAP installation directory
            api_key: ZAP API key
            host: ZAP API host
            port: ZAP API port
        """
        self.zap_path = zap_path or os.environ.get("ZAP_PATH")
        self.api_key = api_key or os.environ.get("ZAP_API_KEY", "")
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/JSON"
        self.scan_id = None
        
        # Make sure ZAP path is found
        if not self.zap_path and not os.environ.get("DISABLE_ZAP_CHECK", ""):
            # Look in standard locations
            potential_paths = [
                "/usr/share/zaproxy",  # Linux
                "/Applications/OWASP ZAP.app/Contents/Java",  # macOS
                r"C:\Program Files\OWASP\Zed Attack Proxy",  # Windows
                r"C:\Program Files (x86)\OWASP\Zed Attack Proxy"  # Windows 32-bit
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    self.zap_path = path
                    break
    
    def start_zap(self, headless: bool = True) -> bool:
        """
        Start ZAP daemon
        
        Args:
            headless: Whether to start ZAP in headless mode
            
        Returns:
            True if ZAP started successfully
        """
        if not self.zap_path:
            logger.error("ZAP path not found. Please set ZAP_PATH environment variable or provide it in the constructor.")
            return False
        
        try:
            # Determine ZAP executable based on OS
            if sys.platform.startswith("win"):
                zap_cmd = os.path.join(self.zap_path, "zap.bat")
            else:
                zap_cmd = os.path.join(self.zap_path, "zap.sh")
            
            # Build command with arguments
            cmd = [zap_cmd, "-daemon"]
            
            if headless:
                cmd.append("-silent")
                
            if self.api_key:
                cmd.extend(["-config", f"api.key={self.api_key}"])
                
            # Start ZAP process
            logger.info(f"Starting ZAP daemon: {' '.join(cmd)}")
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check if ZAP is running by polling the API
            import time
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    response = requests.get(f"{self.api_url}/core/view/version", params={"apikey": self.api_key})
                    if response.status_code == 200:
                        logger.info("ZAP started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                attempts += 1
                time.sleep(5)
                
            logger.error("Failed to start ZAP daemon")
            return False
            
        except Exception as e:
            logger.error(f"Error starting ZAP: {e}")
            return False
    
    def stop_zap(self) -> bool:
        """
        Stop ZAP daemon
        
        Returns:
            True if ZAP stopped successfully
        """
        try:
            response = requests.get(
                f"{self.api_url}/core/action/shutdown", 
                params={"apikey": self.api_key}
            )
            
            if response.status_code == 200:
                logger.info("ZAP shutdown request sent successfully")
                return True
                
            logger.error(f"Error stopping ZAP: {response.text}")
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error stopping ZAP: {e}")
            return False
    
    def _api_request(self, endpoint: str, method: str = "GET", params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the ZAP API
        
        Args:
            endpoint: API endpoint (e.g., "core/view/version")
            method: HTTP method
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.api_url}/{endpoint}"
        
        # Add API key to parameters
        params = params or {}
        if self.api_key:
            params["apikey"] = self.api_key
            
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            if response.status_code == 200:
                return response.json()
                
            logger.error(f"Error calling ZAP API {endpoint}: {response.text}")
            return {"error": f"Status code: {response.status_code}"}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling ZAP API {endpoint}: {e}")
            return {"error": str(e)}
    
    def start_spider(self, target_url: str) -> str:
        """
        Start ZAP spider scan
        
        Args:
            target_url: URL to scan
            
        Returns:
            Scan ID
        """
        logger.info(f"Starting ZAP spider scan for {target_url}")
        
        result = self._api_request(
            "spider/action/scan",
            method="POST",
            params={"url": target_url}
        )
        
        if "error" in result:
            logger.error(f"Failed to start spider scan: {result['error']}")
            return ""
            
        scan_id = result.get("scan", "")
        self.scan_id = scan_id
        logger.info(f"Spider scan started with ID {scan_id}")
        
        return scan_id
    
    def start_active_scan(self, target_url: str) -> str:
        """
        Start ZAP active scan
        
        Args:
            target_url: URL to scan
            
        Returns:
            Scan ID
        """
        logger.info(f"Starting ZAP active scan for {target_url}")
        
        result = self._api_request(
            "ascan/action/scan",
            method="POST",
            params={"url": target_url}
        )
        
        if "error" in result:
            logger.error(f"Failed to start active scan: {result['error']}")
            return ""
            
        scan_id = result.get("scan", "")
        self.scan_id = scan_id
        logger.info(f"Active scan started with ID {scan_id}")
        
        return scan_id
    
    def get_scan_status(self, scan_id: Optional[str] = None, scan_type: str = "active") -> int:
        """
        Get scan status
        
        Args:
            scan_id: Scan ID
            scan_type: Type of scan ('active' or 'spider')
            
        Returns:
            Scan progress percentage
        """
        scan_id = scan_id or self.scan_id
        if not scan_id:
            return 0
            
        endpoint = f"{scan_type}scan/view/status" if scan_type else "ascan/view/status"
        
        result = self._api_request(
            endpoint,
            params={"scanId": scan_id}
        )
        
        if "error" in result:
            logger.error(f"Failed to get scan status: {result['error']}")
            return 0
            
        status = result.get("status", "0")
        try:
            return int(status)
        except ValueError:
            return 0
            
    def wait_for_scan_completion(self, scan_id: Optional[str] = None, scan_type: str = "active", 
                              poll_interval: int = 10, timeout: int = 3600) -> bool:
        """
        Wait for scan to complete
        
        Args:
            scan_id: Scan ID
            scan_type: Type of scan ('active' or 'spider')
            poll_interval: Interval between status checks in seconds
            timeout: Maximum wait time in seconds
            
        Returns:
            True if scan completed, False if timed out
        """
        scan_id = scan_id or self.scan_id
        if not scan_id:
            logger.error("No scan ID provided")
            return False
            
        import time
        start_time = time.time()
        elapsed = 0
        
        logger.info(f"Waiting for {scan_type} scan to complete (timeout: {timeout}s)")
        
        while elapsed < timeout:
            progress = self.get_scan_status(scan_id, scan_type)
            logger.info(f"Scan progress: {progress}%")
            
            if progress >= 100:
                logger.info(f"{scan_type.capitalize()} scan completed successfully")
                return True
                
            time.sleep(poll_interval)
            elapsed = time.time() - start_time
            
        logger.warning(f"{scan_type.capitalize()} scan timed out after {timeout} seconds")
        return False
    
    def get_alerts(self, target_url: Optional[str] = None, risk: Optional[str] = None) -> List[Dict]:
        """
        Get alerts from ZAP
        
        Args:
            target_url: URL to get alerts for
            risk: Risk level to filter by
            
        Returns:
            List of alerts
        """
        params = {}
        if target_url:
            params["url"] = target_url
            
        if risk:
            params["riskId"] = risk
            
        result = self._api_request("core/view/alerts", params=params)
        
        if "error" in result:
            logger.error(f"Failed to get alerts: {result['error']}")
            return []
            
        return result.get("alerts", [])
    
    def generate_report(self, report_path: str, report_format: str = "html") -> bool:
        """
        Generate a report from ZAP
        
        Args:
            report_path: Path to save report to
            report_format: Format of report ('html', 'xml', 'json', or 'md')
            
        Returns:
            True if report was generated successfully
        """
        logger.info(f"Generating {report_format} report at {report_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        
        # Map format to API endpoint
        format_endpoints = {
            "html": "reports/action/generate",
            "xml": "reports/action/generateXml",
            "json": "reports/action/generateJson",
            "md": "reports/action/generateMarkdown"
        }
        
        endpoint = format_endpoints.get(report_format.lower())
        if not endpoint:
            logger.error(f"Unsupported report format: {report_format}")
            return False
            
        result = self._api_request(
            endpoint,
            method="POST",
            params={"path": report_path}
        )
        
        if "error" in result:
            logger.error(f"Failed to generate report: {result['error']}")
            return False
            
        if not os.path.exists(report_path):
            logger.error(f"Report file not created at {report_path}")
            return False
            
        logger.info(f"Report generated successfully at {report_path}")
        return True
    
    def run_full_scan(self, target_url: str) -> SecurityScanResult:
        """
        Run a full ZAP scan (spider + active scan)
        
        Args:
            target_url: URL to scan
            
        Returns:
            SecurityScanResult with scan results
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting full ZAP scan for {target_url}")
        
        # Start spider scan
        spider_id = self.start_spider(target_url)
        if not spider_id:
            logger.error("Failed to start spider scan")
            return SecurityScanResult(
                scan_type="ZAP",
                target=target_url,
                scan_id="failed",
                duration_seconds=0
            )
            
        # Wait for spider to complete
        if not self.wait_for_scan_completion(spider_id, scan_type="spider"):
            logger.warning("Spider scan timed out but continuing with active scan")
            
        # Start active scan
        active_id = self.start_active_scan(target_url)
        if not active_id:
            logger.error("Failed to start active scan")
            return SecurityScanResult(
                scan_type="ZAP",
                target=target_url,
                scan_id=spider_id,
                duration_seconds=time.time() - start_time
            )
            
        # Wait for active scan to complete
        if not self.wait_for_scan_completion(active_id, scan_type="active"):
            logger.warning("Active scan timed out")
            
        # Get alerts
        alerts = self.get_alerts(target_url)
        
        # Process alerts into vulnerabilities
        vulnerabilities = []
        severity_counts = {"INFO": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        
        for alert in alerts:
            severity = alert.get("risk", "").upper()
            if severity == "INFORMATIONAL":
                severity = "INFO"
            elif severity == "LOW":
                severity = "LOW"
            elif severity == "MEDIUM":
                severity = "MEDIUM"
            elif severity == "HIGH":
                severity = "HIGH"
            else:
                severity = "INFO"
                
            vulnerability = SecurityVulnerability(
                id=alert.get("alertRef", ""),
                name=alert.get("name", ""),
                description=alert.get("description", ""),
                severity=severity,
                evidence=alert.get("evidence", ""),
                url=alert.get("url", ""),
                param=alert.get("param", ""),
                attack=alert.get("attack", ""),
                solution=alert.get("solution", ""),
                cwe_id=int(alert.get("cweid", "0")) if alert.get("cweid") else None,
                wasc_id=int(alert.get("wascid", "0")) if alert.get("wascid") else None,
                source="ZAP"
            )
            
            vulnerabilities.append(vulnerability)
            severity_counts[severity] += 1
            
        # Create scan result
        result = SecurityScanResult(
            scan_type="ZAP",
            target=target_url,
            total_vulnerabilities=len(vulnerabilities),
            vulnerabilities_by_severity=severity_counts,
            vulnerabilities=vulnerabilities,
            scan_id=active_id,
            duration_seconds=time.time() - start_time
        )
        
        logger.info(f"Scan completed with {len(vulnerabilities)} vulnerabilities found")
        for sev, count in severity_counts.items():
            if count > 0:
                logger.info(f"  {sev}: {count}")
                
        return result
        
    def save_scan_results(self, results: SecurityScanResult, output_path: str) -> bool:
        """
        Save scan results to file
        
        Args:
            results: Scan results
            output_path: Path to save results to
            
        Returns:
            True if results saved successfully
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write results to file
            with open(output_path, "w") as f:
                f.write(results.json(indent=2))
                
            logger.info(f"Scan results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving scan results: {e}")
            return False


class OWASPDependencyCheck:
    """
    OWASP Dependency Check integration
    
    This class provides methods to scan project dependencies for vulnerabilities.
    """
    
    def __init__(self, dc_path: Optional[str] = None):
        """
        Initialize Dependency Check
        
        Args:
            dc_path: Path to Dependency Check installation directory
        """
        self.dc_path = dc_path or os.environ.get("DEPENDENCY_CHECK_PATH")
        
        # Check if path is valid
        if self.dc_path and not os.path.exists(self.dc_path):
            logger.warning(f"Dependency Check path not found: {self.dc_path}")
            self.dc_path = None
            
        # Look for dependency-check in PATH
        if not self.dc_path:
            try:
                # Check if installed in PATH
                subprocess.run(
                    ["dependency-check", "--version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    check=False
                )
                self.use_system_install = True
            except (subprocess.SubprocessError, FileNotFoundError):
                self.use_system_install = False
                
    def scan_dependencies(self, 
                       project_path: str, 
                       output_path: str, 
                       report_format: str = "JSON",
                       scan_python: bool = True,
                       scan_js: bool = True) -> bool:
        """
        Scan project dependencies for vulnerabilities
        
        Args:
            project_path: Path to project to scan
            output_path: Path to save report to
            report_format: Format of report (HTML, XML, JSON, CSV)
            scan_python: Whether to scan Python dependencies
            scan_js: Whether to scan JavaScript dependencies
            
        Returns:
            True if scan completed successfully
        """
        logger.info(f"Scanning dependencies for {project_path}")
        
        # Determine command to run
        if self.use_system_install:
            cmd = ["dependency-check"]
        elif self.dc_path:
            if sys.platform.startswith("win"):
                cmd = [os.path.join(self.dc_path, "bin", "dependency-check.bat")]
            else:
                cmd = [os.path.join(self.dc_path, "bin", "dependency-check.sh")]
        else:
            logger.error("Dependency Check not found. Install it or set DEPENDENCY_CHECK_PATH environment variable.")
            return False
            
        # Add command arguments
        cmd.extend(["--project", os.path.basename(project_path)])
        cmd.extend(["--scan", project_path])
        cmd.extend(["--out", output_path])
        cmd.extend(["--format", report_format])
        
        # Configure scanners
        scanners = []
        if scan_python:
            scanners.append("PYTHON")
        if scan_js:
            scanners.extend(["NODEJS", "JAVASCRIPT", "JAVASCRIPT_YARN", "JAVASCRIPT_NPM"])
        
        if scanners:
            cmd.extend(["--enabledSearchers", ",".join(scanners)])
        
        # Run the scan
        try:
            logger.info(f"Running Dependency Check: {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                logger.error(f"Dependency Check failed with exit code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                return False
                
            logger.info("Dependency Check completed successfully")
            
            # Check if report was created
            if not os.path.exists(output_path):
                logger.error(f"Report file not created at {output_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error running Dependency Check: {e}")
            return False
    
    def parse_report(self, report_path: str) -> SecurityScanResult:
        """
        Parse Dependency Check report
        
        Args:
            report_path: Path to report file
            
        Returns:
            SecurityScanResult with parsed vulnerabilities
        """
        try:
            with open(report_path, "r") as f:
                report_data = json.load(f)
                
            # Extract basic scan information
            scan_info = report_data.get("scanInfo", {})
            project_info = report_data.get("projectInfo", {})
            
            # Extract dependencies
            dependencies = report_data.get("dependencies", [])
            
            # Process vulnerabilities
            vulnerabilities = []
            severity_counts = {"INFO": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            
            for dep in dependencies:
                # Skip dependencies without vulnerabilities
                if not dep.get("vulnerabilities"):
                    continue
                    
                dep_name = dep.get("fileName", "unknown")
                
                for vuln in dep.get("vulnerabilities", []):
                    # Determine severity
                    cvss_v3 = vuln.get("cvssv3", {})
                    cvss_v2 = vuln.get("cvssv2", {})
                    
                    # Try to get severity from CVSS score
                    score = cvss_v3.get("baseScore", cvss_v2.get("score", 0))
                    
                    if score >= 9.0:
                        severity = "CRITICAL"
                    elif score >= 7.0:
                        severity = "HIGH"
                    elif score >= 4.0:
                        severity = "MEDIUM"
                    elif score > 0:
                        severity = "LOW"
                    else:
                        severity = "INFO"
                        
                    # Create vulnerability object
                    vulnerability = SecurityVulnerability(
                        id=vuln.get("name", ""),
                        name=f"{dep_name}: {vuln.get('name', '')}",
                        description=vuln.get("description", ""),
                        severity=severity,
                        url=next(iter(vuln.get("references", [])), {}).get("url", ""),
                        cwe_id=next((int(cwe.replace("CWE-", "")) for cwe in vuln.get("cwes", []) if cwe.startswith("CWE-")), None),
                        source="Dependency-Check"
                    )
                    
                    vulnerabilities.append(vulnerability)
                    severity_counts[severity] += 1
            
            # Create scan result
            result = SecurityScanResult(
                scan_type="Dependency-Check",
                target=project_info.get("name", "unknown"),
                total_vulnerabilities=len(vulnerabilities),
                vulnerabilities_by_severity=severity_counts,
                vulnerabilities=vulnerabilities,
                scan_id=scan_info.get("reportId", ""),
                duration_seconds=float(scan_info.get("engineSeconds", 0))
            )
            
            logger.info(f"Parsed report with {len(vulnerabilities)} vulnerabilities found")
            for sev, count in severity_counts.items():
                if count > 0:
                    logger.info(f"  {sev}: {count}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Dependency Check report: {e}")
            return SecurityScanResult(
                scan_type="Dependency-Check",
                target="unknown",
                scan_id="error",
                duration_seconds=0
            )


def setup_ci_security_testing() -> Dict[str, Any]:
    """
    Setup CI security testing
    
    Returns:
        Dictionary with CI setup results
    """
    results = {}
    
    # Create security testing directories
    for directory in ["reports/security", "logs/security"]:
        os.makedirs(directory, exist_ok=True)
        
    # Setup ZAP
    zap_env_file = os.path.join(".github", "workflows", "zap-env.txt")
    
    if not os.path.exists(os.path.dirname(zap_env_file)):
        os.makedirs(os.path.dirname(zap_env_file), exist_ok=True)
        
    with open(zap_env_file, "w") as f:
        f.write("ZAP_API_KEY=" + os.urandom(16).hex() + "\n")
        
    results["zap_env_file_created"] = True
    
    # Create GitHub workflow file for ZAP scanning
    workflow_content = """name: Security Testing

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  schedule:
    - cron: '0 2 * * 1'  # Every Monday at 2 AM

jobs:
  zap_scan:
    runs-on: ubuntu-latest
    name: OWASP ZAP API Scan
    steps:
      - uses: actions/checkout@v3
      
      - name: ZAP Scan
        uses: zaproxy/action-api-scan@v0.5.0
        with:
          target: 'https://staging.ai-trading-agent.com/api'
          fail_action: false
          allow_issue_writing: true
          issue_title: 'ZAP Scan Report'
          token: ${{ github.token }}
          
  dependency_check:
    runs-on: ubuntu-latest
    name: OWASP Dependency Check
    steps:
      - uses: actions/checkout@v3
      
      - name: Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'AI Trading Agent'
          path: '.'
          format: 'HTML'
          out: 'reports/security/dependency-check-report.html'
          args: >
            --enableExperimental
            --scan **/*.py
            --scan **/*.js
            --scan **/requirements.txt
            --scan **/package.json
      
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: Dependency Check Report
          path: reports/security/dependency-check-report.html
          
  secrets_scanning:
    runs-on: ubuntu-latest
    name: Secret Scanning
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          
      - name: TruffleHog Scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
          extra_args: --debug --only-verified
"""
    
    workflow_path = os.path.join(".github", "workflows", "security-testing.yml")
    
    if not os.path.exists(os.path.dirname(workflow_path)):
        os.makedirs(os.path.dirname(workflow_path), exist_ok=True)
        
    with open(workflow_path, "w") as f:
        f.write(workflow_content)
        
    results["workflow_file_created"] = True
    
    return results


def run_local_security_scan(target_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Run a local security scan
    
    Args:
        target_url: URL to scan
        
    Returns:
        Dictionary with scan results
    """
    results = {}
    
    # Create output directories
    for directory in ["reports/security", "logs/security"]:
        os.makedirs(directory, exist_ok=True)
    
    # Run ZAP scan if available
    zap_scanner = ZAPScanner()
    if zap_scanner.zap_path or os.environ.get("ZAP_PATH"):
        try:
            # Start ZAP
            if zap_scanner.start_zap():
                # Run scan
                scan_result = zap_scanner.run_full_scan(target_url)
                
                # Save results
                results_path = "reports/security/zap-scan-results.json"
                zap_scanner.save_scan_results(scan_result, results_path)
                
                # Generate report
                report_path = "reports/security/zap-scan-report.html"
                zap_scanner.generate_report(report_path)
                
                # Stop ZAP
                zap_scanner.stop_zap()
                
                results["zap_scan"] = {
                    "status": "completed",
                    "vulnerabilities": scan_result.total_vulnerabilities,
                    "by_severity": scan_result.vulnerabilities_by_severity,
                    "report_path": report_path,
                    "results_path": results_path
                }
            else:
                results["zap_scan"] = {
                    "status": "failed",
                    "error": "Failed to start ZAP"
                }
        except Exception as e:
            results["zap_scan"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        results["zap_scan"] = {
            "status": "skipped",
            "reason": "ZAP not found"
        }
        
    # Run Dependency Check if available
    dc_scanner = OWASPDependencyCheck()
    if dc_scanner.dc_path or dc_scanner.use_system_install:
        try:
            # Run scan
            report_path = "reports/security/dependency-check-report.json"
            if dc_scanner.scan_dependencies(".", report_path):
                # Parse report
                scan_result = dc_scanner.parse_report(report_path)
                
                results["dependency_check"] = {
                    "status": "completed",
                    "vulnerabilities": scan_result.total_vulnerabilities,
                    "by_severity": scan_result.vulnerabilities_by_severity,
                    "report_path": report_path
                }
            else:
                results["dependency_check"] = {
                    "status": "failed",
                    "error": "Scan failed"
                }
        except Exception as e:
            results["dependency_check"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        results["dependency_check"] = {
            "status": "skipped",
            "reason": "Dependency Check not found"
        }
        
    return results
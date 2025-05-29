import logging
import random
import time
from typing import Dict, List, Any, Optional
from kubernetes.client.rest import ApiException
from base import BaseChaosTest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceExhaustionTest(BaseChaosTest):
    """Test system recovery from resource (CPU/memory) exhaustion."""
    
    def __init__(self, 
                 namespace: str = "ai-trading", 
                 recovery_timeout: int = 300,
                 target_service: str = "api",
                 resource_type: str = "cpu",  # "cpu" or "memory"
                 exhaustion_duration: int = 120):
        """
        Initialize the resource exhaustion test.
        
        Args:
            namespace: Kubernetes namespace where the system is deployed
            recovery_timeout: Maximum time in seconds to wait for recovery
            target_service: Service to exhaust resources on
            resource_type: Type of resource to exhaust ("cpu" or "memory")
            exhaustion_duration: Duration to maintain resource exhaustion in seconds
        """
        super().__init__(namespace, recovery_timeout)
        self.target_service = target_service
        self.resource_type = resource_type.lower()
        self.exhaustion_duration = exhaustion_duration
        self.target_pods = []
        self.stress_processes = {}
        
        # Validate resource type
        if self.resource_type not in ["cpu", "memory"]:
            raise ValueError("resource_type must be either 'cpu' or 'memory'")
        
    def setup(self):
        """Set up the test by identifying target pods."""
        # Get all pods in the namespace
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        
        # Find target pods
        for pod in pods.items:
            if pod.status.phase == "Running":
                # Match by label or name
                if (
                    self.target_service in pod.metadata.name or 
                    pod.metadata.labels.get("app") == self.target_service
                ):
                    # Skip trading-agent pod as it's critical
                    if "trading-agent" not in pod.metadata.name:
                        self.target_pods.append(pod)
        
        # Ensure we have pods to target
        if not self.target_pods:
            raise ValueError(f"No pods found for target service: {self.target_service}")
            
        logger.info(f"Found {len(self.target_pods)} target pods for resource exhaustion")

        # Check if stress tool is available in pods
        for pod in self.target_pods:
            try:
                # Check if stress-ng is installed
                exec_command = ["/bin/sh", "-c", "which stress-ng || echo 'not found'"]
                
                resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                    name=pod.metadata.name,
                    namespace=self.namespace,
                    command=exec_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                
                if "not found" in resp:
                    logger.warning(f"stress-ng not found in pod {pod.metadata.name}, trying to install...")
                    
                    # Try to install stress-ng
                    install_cmd = [
                        "/bin/sh", 
                        "-c", 
                        "apt-get update && apt-get install -y stress-ng || " + 
                        "apk add --no-cache stress-ng || " + 
                        "yum install -y stress-ng || " +
                        "echo 'Failed to install stress-ng'"
                    ]
                    
                    install_resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                        name=pod.metadata.name,
                        namespace=self.namespace,
                        command=install_cmd,
                        stderr=True,
                        stdin=False,
                        stdout=True,
                        tty=False
                    )
                    
                    logger.info(f"Installation result: {install_resp}")
                    
            except ApiException as e:
                logger.error(f"Error checking/installing stress tools in pod {pod.metadata.name}: {e}")
        
    def execute(self):
        """Execute resource exhaustion by stressing CPU or memory."""
        for pod in self.target_pods:
            try:
                logger.info(f"Initiating {self.resource_type} exhaustion for pod {pod.metadata.name}")
                
                # Determine resource limits for the pod
                resource_limit = None
                for container in pod.spec.containers:
                    if container.resources and container.resources.limits:
                        if self.resource_type == "cpu":
                            cpu_limit = container.resources.limits.get("cpu", "1")
                            # Convert from Kubernetes CPU units to cores
                            if cpu_limit.endswith("m"):
                                resource_limit = int(cpu_limit[:-1]) / 1000
                            else:
                                resource_limit = int(cpu_limit)
                        else:  # memory
                            mem_limit = container.resources.limits.get("memory", "1Gi")
                            # Convert from Kubernetes memory units to MB
                            if mem_limit.endswith("Gi"):
                                resource_limit = int(float(mem_limit[:-2]) * 1024)
                            elif mem_limit.endswith("Mi"):
                                resource_limit = int(mem_limit[:-2])
                            else:
                                resource_limit = 1024  # Default 1GB
                
                # If no limit found, use safe defaults
                if not resource_limit:
                    resource_limit = 1 if self.resource_type == "cpu" else 1024
                
                # Create stress command based on resource type
                if self.resource_type == "cpu":
                    # Use 80% of available CPUs
                    cpu_count = max(1, int(resource_limit * 0.8))
                    stress_cmd = f"stress-ng --cpu {cpu_count} --cpu-method all --timeout {self.exhaustion_duration}s"
                else:  # memory
                    # Use 80% of available memory
                    mem_mb = max(128, int(resource_limit * 0.8))
                    stress_cmd = f"stress-ng --vm 1 --vm-bytes {mem_mb}M --timeout {self.exhaustion_duration}s"
                
                # Execute the stress command in background
                exec_command = [
                    "/bin/sh", 
                    "-c", 
                    f"nohup {stress_cmd} > /tmp/stress.out 2>&1 & echo $!"
                ]
                
                resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                    name=pod.metadata.name,
                    namespace=self.namespace,
                    command=exec_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                
                # Store the process ID for later cleanup
                try:
                    process_id = int(resp.strip())
                    self.stress_processes[pod.metadata.name] = process_id
                    logger.info(f"Started stress process {process_id} on pod {pod.metadata.name}")
                except ValueError:
                    logger.error(f"Failed to parse process ID from: {resp}")
                
            except ApiException as e:
                logger.error(f"Error creating resource exhaustion for pod {pod.metadata.name}: {e}")
        
        logger.info(f"Created {self.resource_type} exhaustion on {len(self.stress_processes)} pods")
        logger.info(f"Maintaining resource exhaustion for {self.exhaustion_duration} seconds")
        
        # Wait for a portion of the exhaustion duration to allow monitoring
        # The processes will continue running for the full duration
        wait_time = min(30, self.exhaustion_duration * 0.25)
        time.sleep(wait_time)
    
    def verify(self) -> bool:
        """
        Verify the system handled resource exhaustion correctly.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        # Check that throttling is happening as expected
        for pod_name in self.stress_processes:
            try:
                # Check pod metrics
                if self.resource_type == "cpu":
                    exec_command = [
                        "/bin/sh", 
                        "-c", 
                        "cat /proc/stat | grep '^cpu '"
                    ]
                else:  # memory
                    exec_command = [
                        "/bin/sh", 
                        "-c", 
                        "cat /proc/meminfo | grep 'MemAvailable'"
                    ]
                
                resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                    name=pod_name,
                    namespace=self.namespace,
                    command=exec_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                
                logger.info(f"Resource usage on pod {pod_name}: {resp}")
                
            except ApiException as e:
                logger.error(f"Error checking resource usage on pod {pod_name}: {e}")
        
        # Wait for the exhaustion duration to complete
        remaining_time = self.exhaustion_duration - 30
        if remaining_time > 0:
            logger.info(f"Waiting for stress test to complete, {remaining_time} seconds remaining...")
            time.sleep(remaining_time)
        
        # Wait additional time for recovery
        logger.info("Waiting for system recovery...")
        time.sleep(30)
        
        # Check that the system is still operational after resource exhaustion
        try:
            # Get deployment status
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
            
            all_healthy = True
            for deployment in deployments.items:
                if deployment.spec.replicas != deployment.status.ready_replicas:
                    logger.error(
                        f"Deployment {deployment.metadata.name} has {deployment.status.ready_replicas}/{deployment.spec.replicas} "
                        f"ready replicas after resource exhaustion test"
                    )
                    all_healthy = False
            
            return all_healthy
            
        except ApiException as e:
            logger.error(f"Error during verification: {e}")
            return False
    
    def cleanup(self):
        """Clean up by stopping stress processes."""
        for pod_name, process_id in self.stress_processes.items():
            try:
                logger.info(f"Stopping stress process {process_id} on pod {pod_name}")
                
                # Kill the stress process
                exec_command = [
                    "/bin/sh", 
                    "-c", 
                    f"kill -9 {process_id} || true"
                ]
                
                resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                    name=pod_name,
                    namespace=self.namespace,
                    command=exec_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False
                )
                
                logger.info(f"Process termination result: {resp}")
                
            except ApiException as e:
                logger.error(f"Error stopping stress process on pod {pod_name}: {e}")

def test_cpu_exhaustion():
    """Run the CPU resource exhaustion test as a pytest function."""
    test = ResourceExhaustionTest(
        target_service="api",
        resource_type="cpu",
        exhaustion_duration=120
    )
    assert test.run() is True

def test_memory_exhaustion():
    """Run the memory resource exhaustion test as a pytest function."""
    test = ResourceExhaustionTest(
        target_service="api",
        resource_type="memory",
        exhaustion_duration=120
    )
    assert test.run() is True

if __name__ == "__main__":
    # Run directly
    test = ResourceExhaustionTest(resource_type="cpu")
    test.run()

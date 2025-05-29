import logging
import random
import time
from typing import Dict, List, Any, Optional
from kubernetes.client.rest import ApiException
from base import BaseChaosTest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkDelayTest(BaseChaosTest):
    """Test system recovery from network delays between components."""
    
    def __init__(self, 
                 namespace: str = "ai-trading", 
                 recovery_timeout: int = 300,
                 target_services: Optional[List[str]] = None,
                 delay_ms: int = 200,
                 delay_duration: int = 60):
        """
        Initialize the network delay test.
        
        Args:
            namespace: Kubernetes namespace where the system is deployed
            recovery_timeout: Maximum time in seconds to wait for recovery
            target_services: List of service names to target (if None, chooses randomly)
            delay_ms: Network delay to inject in milliseconds
            delay_duration: Duration to maintain the network delay in seconds
        """
        super().__init__(namespace, recovery_timeout)
        self.target_services = target_services or ["api", "redis", "postgres"]
        self.delay_ms = delay_ms
        self.delay_duration = delay_duration
        self.affected_pods = []
        
    def setup(self):
        """Set up the test by selecting target pods."""
        # Get all pods corresponding to the target services
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        
        # Filter pods by service
        available_pods = []
        for pod in pods.items:
            if pod.status.phase == "Running":
                for service in self.target_services:
                    # Match by label or name
                    if (
                        service in pod.metadata.name or 
                        pod.metadata.labels.get("app") == service
                    ):
                        available_pods.append(pod)
                        break
        
        # Ensure we have pods to target
        if not available_pods:
            raise ValueError(f"No pods found for services: {self.target_services}")
            
        self.target_pods = available_pods
        logger.info(f"Selected {len(self.target_pods)} pods for network delay injection")
        
    def execute(self):
        """Execute the network delay injection."""
        for pod in self.target_pods:
            try:
                logger.info(f"Injecting {self.delay_ms}ms network delay to pod {pod.metadata.name}")
                
                # Use a Kubernetes exec command to run tc (traffic control) inside the pod
                # Note: This requires the pod to have the necessary privileges and tools
                exec_command = [
                    "/bin/sh", 
                    "-c", 
                    f"tc qdisc add dev eth0 root netem delay {self.delay_ms}ms"
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
                
                logger.info(f"Network delay injection result: {resp}")
                self.affected_pods.append(pod.metadata.name)
                
            except ApiException as e:
                logger.error(f"Error injecting network delay to pod {pod.metadata.name}: {e}")
        
        logger.info(f"Injected network delay to {len(self.affected_pods)} pods")
        logger.info(f"Maintaining network delay for {self.delay_duration} seconds")
        time.sleep(self.delay_duration)
    
    def verify(self) -> bool:
        """
        Verify the system recovered correctly.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        # Check that the system is still operational despite delays
        try:
            # Get deployment status
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
            
            for deployment in deployments.items:
                if deployment.status.unavailable_replicas and deployment.status.unavailable_replicas > 0:
                    logger.error(
                        f"Deployment {deployment.metadata.name} has {deployment.status.unavailable_replicas} "
                        f"unavailable replicas during network delay test"
                    )
                    return False
            
            # Check service connectivity (we could add more specific checks here)
            # For example, make API calls to verify functionality
            
            # If we made it here, verification passed
            return True
            
        except ApiException as e:
            logger.error(f"Error during verification: {e}")
            return False
    
    def cleanup(self):
        """Clean up by removing network delays."""
        for pod_name in self.affected_pods:
            try:
                logger.info(f"Removing network delay from pod {pod_name}")
                
                exec_command = [
                    "/bin/sh", 
                    "-c", 
                    "tc qdisc del dev eth0 root"
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
                
                logger.info(f"Network delay removal result: {resp}")
                
            except ApiException as e:
                logger.error(f"Error removing network delay from pod {pod_name}: {e}")

def test_network_delay():
    """Run the network delay test as a pytest function."""
    test = NetworkDelayTest(
        target_services=["api", "redis"],
        delay_ms=200,
        delay_duration=60
    )
    assert test.run() is True

if __name__ == "__main__":
    # Run directly
    test = NetworkDelayTest()
    test.run()

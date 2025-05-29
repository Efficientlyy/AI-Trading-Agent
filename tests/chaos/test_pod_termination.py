import logging
import random
import time
from typing import Dict, List, Any, Optional

from kubernetes.client.rest import ApiException
from base import BaseChaosTest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PodTerminationTest(BaseChaosTest):
    """Test system recovery from pod terminations."""
    
    def __init__(self, 
                 namespace: str = "ai-trading", 
                 recovery_timeout: int = 300,
                 target_deployments: Optional[List[str]] = None,
                 termination_count: int = 1):
        """
        Initialize the pod termination test.
        
        Args:
            namespace: Kubernetes namespace where the system is deployed
            recovery_timeout: Maximum time in seconds to wait for recovery
            target_deployments: List of deployment names to target (if None, chooses randomly)
            termination_count: Number of pods to terminate
        """
        super().__init__(namespace, recovery_timeout)
        self.target_deployments = target_deployments
        self.termination_count = termination_count
        self.terminated_pods = []
        
    def setup(self):
        """Set up the test by selecting target pods."""
        # Get all pods in the namespace
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        
        # Filter by deployment if specified
        available_pods = []
        for pod in pods.items:
            if pod.status.phase == "Running":
                if self.target_deployments is None or any(
                    td in pod.metadata.name for td in self.target_deployments
                ):
                    # Don't terminate trading-agent pod as it's designed to be a singleton
                    if "trading-agent" not in pod.metadata.name:
                        available_pods.append(pod)
        
        # Ensure we have enough pods
        if len(available_pods) < self.termination_count:
            raise ValueError(
                f"Not enough available pods ({len(available_pods)}) "
                f"to terminate {self.termination_count}"
            )
            
        # Randomly select pods to terminate
        self.target_pods = random.sample(available_pods, self.termination_count)
        logger.info(f"Selected {len(self.target_pods)} pods for termination")
        
    def execute(self):
        """Execute the pod termination."""
        for pod in self.target_pods:
            try:
                logger.info(f"Terminating pod {pod.metadata.name}")
                self.k8s_core_v1.delete_namespaced_pod(
                    name=pod.metadata.name,
                    namespace=self.namespace,
                    body={}
                )
                self.terminated_pods.append(pod.metadata.name)
            except ApiException as e:
                logger.error(f"Error terminating pod {pod.metadata.name}: {e}")
                
        logger.info(f"Terminated {len(self.terminated_pods)} pods")
    
    def verify(self) -> bool:
        """
        Verify the system recovered correctly.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        # Check that new pods were created to replace the terminated ones
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        
        # Check deployments have correct replica counts
        deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
        
        for deployment in deployments.items:
            original_replicas = self.original_state["deployments"].get(deployment.metadata.name)
            if original_replicas is not None and deployment.status.ready_replicas != original_replicas:
                logger.error(
                    f"Deployment {deployment.metadata.name} has {deployment.status.ready_replicas} "
                    f"ready replicas, expected {original_replicas}"
                )
                return False
                
        # All checks passed
        return True
    
    def cleanup(self):
        """Clean up after the test (no specific cleanup needed)."""
        pass

def test_pod_termination():
    """Run the pod termination test as a pytest function."""
    test = PodTerminationTest(
        target_deployments=["api", "dashboard"],
        termination_count=1
    )
    assert test.run() is True

if __name__ == "__main__":
    # Run directly
    test = PodTerminationTest()
    test.run()

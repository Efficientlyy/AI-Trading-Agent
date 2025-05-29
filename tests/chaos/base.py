import abc
import logging
import time
from typing import Dict, List, Any, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseChaosTest(abc.ABC):
    """Base class for all chaos tests."""
    
    def __init__(self, namespace: str = "ai-trading", recovery_timeout: int = 300):
        """
        Initialize the chaos test.
        
        Args:
            namespace: Kubernetes namespace where the system is deployed
            recovery_timeout: Maximum time in seconds to wait for recovery
        """
        self.namespace = namespace
        self.recovery_timeout = recovery_timeout
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        self.original_state = {}
        self.chaos_start_time = None
        self.test_name = self.__class__.__name__
        
    def initialize_kubernetes_client(self):
        """Initialize the Kubernetes client."""
        try:
            config.load_kube_config()
        except:
            # Fallback to in-cluster config when running in a pod
            config.load_incluster_config()
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
    
    def record_system_state(self) -> Dict[str, Any]:
        """Record the current state of the system for comparison after recovery."""
        deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        services = self.k8s_core_v1.list_namespaced_service(namespace=self.namespace)
        
        state = {
            "deployments": {d.metadata.name: d.status.available_replicas for d in deployments.items},
            "pods": {p.metadata.name: p.status.phase for p in pods.items},
            "services": {s.metadata.name: s.spec.type for s in services.items}
        }
        
        return state
    
    def wait_for_recovery(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for the system to recover from chaos.
        
        Args:
            timeout: Override the default recovery timeout
            
        Returns:
            bool: True if the system recovered, False otherwise
        """
        if timeout is None:
            timeout = self.recovery_timeout
            
        logger.info(f"Waiting up to {timeout} seconds for system to recover...")
        
        end_time = time.time() + timeout
        recovered = False
        
        while time.time() < end_time and not recovered:
            try:
                # Check that all deployments have expected replica count
                deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
                all_ready = True
                
                for d in deployments.items:
                    if d.status.ready_replicas != d.status.replicas:
                        all_ready = False
                        logger.info(f"Deployment {d.metadata.name} has {d.status.ready_replicas}/{d.status.replicas} ready replicas")
                        break
                
                if all_ready:
                    # Additional health checks could be added here
                    recovered = True
                    break
                    
            except ApiException as e:
                logger.error(f"Error checking recovery status: {e}")
                
            time.sleep(5)
        
        if recovered:
            recovery_time = time.time() - self.chaos_start_time
            logger.info(f"System recovered after {recovery_time:.2f} seconds")
        else:
            logger.error("System failed to recover within timeout period")
            
        return recovered
        
    def run(self) -> bool:
        """
        Run the chaos test.
        
        Returns:
            bool: True if the test passed, False otherwise
        """
        logger.info(f"Starting chaos test: {self.test_name}")
        
        try:
            # Initialize the Kubernetes client
            self.initialize_kubernetes_client()
            
            # Setup the test
            logger.info("Setting up test...")
            self.setup()
            
            # Record the original state
            self.original_state = self.record_system_state()
            
            # Execute chaos
            logger.info("Executing chaos...")
            self.chaos_start_time = time.time()
            self.execute()
            
            # Wait for recovery
            recovered = self.wait_for_recovery()
            
            # Verify the system state
            if recovered:
                logger.info("Verifying system state...")
                verified = self.verify()
            else:
                verified = False
                
            # Clean up
            logger.info("Cleaning up...")
            self.cleanup()
            
            # Determine test result
            if recovered and verified:
                logger.info(f"Chaos test {self.test_name} PASSED")
                return True
            else:
                logger.error(f"Chaos test {self.test_name} FAILED")
                return False
                
        except Exception as e:
            logger.exception(f"Error during chaos test: {e}")
            return False
    
    @abc.abstractmethod
    def setup(self):
        """Set up the test. Must be implemented by subclasses."""
        pass
    
    @abc.abstractmethod
    def execute(self):
        """Execute the chaos operation. Must be implemented by subclasses."""
        pass
    
    @abc.abstractmethod
    def verify(self) -> bool:
        """
        Verify the system recovered correctly. Must be implemented by subclasses.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self):
        """Clean up after the test. Must be implemented by subclasses."""
        pass

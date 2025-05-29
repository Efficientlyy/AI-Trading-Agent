import logging
import random
import time
from typing import Dict, List, Any, Optional
from kubernetes.client.rest import ApiException
from base import BaseChaosTest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkPartitionTest(BaseChaosTest):
    """Test system recovery from network partitions between services."""
    
    def __init__(self, 
                 namespace: str = "ai-trading", 
                 recovery_timeout: int = 300,
                 source_service: str = "api",
                 target_service: str = "redis",
                 partition_duration: int = 60):
        """
        Initialize the network partition test.
        
        Args:
            namespace: Kubernetes namespace where the system is deployed
            recovery_timeout: Maximum time in seconds to wait for recovery
            source_service: Service that will be isolated from target_service
            target_service: Service that will be isolated from source_service
            partition_duration: Duration to maintain the network partition in seconds
        """
        super().__init__(namespace, recovery_timeout)
        self.source_service = source_service
        self.target_service = target_service
        self.partition_duration = partition_duration
        self.source_pods = []
        self.target_pods = []
        self.affected_pods = []
        
    def setup(self):
        """Set up the test by identifying source and target pods."""
        # Get all pods in the namespace
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        
        # Find source and target pods
        for pod in pods.items:
            if pod.status.phase == "Running":
                # Match by label or name
                if (
                    self.source_service in pod.metadata.name or 
                    pod.metadata.labels.get("app") == self.source_service
                ):
                    self.source_pods.append(pod)
                    
                if (
                    self.target_service in pod.metadata.name or 
                    pod.metadata.labels.get("app") == self.target_service
                ):
                    self.target_pods.append(pod)
        
        # Ensure we have pods to target
        if not self.source_pods:
            raise ValueError(f"No pods found for source service: {self.source_service}")
            
        if not self.target_pods:
            raise ValueError(f"No pods found for target service: {self.target_service}")
            
        logger.info(f"Found {len(self.source_pods)} source pods and {len(self.target_pods)} target pods")
        
    def execute(self):
        """Execute the network partition by blocking traffic between services."""
        # Get target pod IPs
        target_ips = []
        for pod in self.target_pods:
            if pod.status.pod_ip:
                target_ips.append(pod.status.pod_ip)
        
        logger.info(f"Target service IPs: {target_ips}")
        
        # Block traffic from source pods to target IPs
        for pod in self.source_pods:
            try:
                logger.info(f"Creating network partition for pod {pod.metadata.name}")
                
                for target_ip in target_ips:
                    # Use iptables to block traffic to target pods
                    exec_command = [
                        "/bin/sh", 
                        "-c", 
                        f"iptables -A OUTPUT -d {target_ip} -j DROP"
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
                    
                    logger.info(f"Network partition command result: {resp}")
                    
                self.affected_pods.append(pod.metadata.name)
                
            except ApiException as e:
                logger.error(f"Error creating network partition for pod {pod.metadata.name}: {e}")
        
        logger.info(f"Created network partition between {len(self.affected_pods)} source pods and {len(target_ips)} target IPs")
        logger.info(f"Maintaining network partition for {self.partition_duration} seconds")
        time.sleep(self.partition_duration)
    
    def verify(self) -> bool:
        """
        Verify the system recovered correctly after network partition.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        # Wait for recovery after removing network partition
        time.sleep(10)  # Give system time to reconnect
        
        try:
            # Get current system state
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
            
            # Check all deployments are fully available
            for deployment in deployments.items:
                if (
                    deployment.status.available_replicas != deployment.status.replicas or 
                    deployment.status.unavailable_replicas
                ):
                    logger.error(
                        f"Deployment {deployment.metadata.name} has not recovered properly. "
                        f"Available: {deployment.status.available_replicas}/{deployment.status.replicas}, "
                        f"Unavailable: {deployment.status.unavailable_replicas or 0}"
                    )
                    return False
            
            # If we made it here, verification passed
            return True
            
        except ApiException as e:
            logger.error(f"Error during verification: {e}")
            return False
    
    def cleanup(self):
        """Clean up by removing network partition rules."""
        # Get target pod IPs again (in case they changed)
        target_ips = []
        for pod in self.target_pods:
            if pod.status.pod_ip:
                target_ips.append(pod.status.pod_ip)
        
        # Remove iptables rules from source pods
        for pod_name in self.affected_pods:
            try:
                logger.info(f"Removing network partition for pod {pod_name}")
                
                for target_ip in target_ips:
                    # Remove the iptables rules
                    exec_command = [
                        "/bin/sh", 
                        "-c", 
                        f"iptables -D OUTPUT -d {target_ip} -j DROP"
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
                    
                    logger.info(f"Network partition removal result: {resp}")
                    
            except ApiException as e:
                logger.error(f"Error removing network partition for pod {pod_name}: {e}")

def test_network_partition():
    """Run the network partition test as a pytest function."""
    test = NetworkPartitionTest(
        source_service="api",
        target_service="redis",
        partition_duration=60
    )
    assert test.run() is True

if __name__ == "__main__":
    # Run directly
    test = NetworkPartitionTest()
    test.run()

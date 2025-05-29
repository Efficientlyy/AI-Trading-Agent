import logging
import random
import time
from typing import Dict, List, Any, Optional
from kubernetes.client.rest import ApiException
from base import BaseChaosTest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseFailureTest(BaseChaosTest):
    """Test system recovery from database failures."""
    
    def __init__(self, 
                 namespace: str = "ai-trading", 
                 recovery_timeout: int = 300,
                 database_service: str = "postgres",
                 failure_duration: int = 60,
                 failure_type: str = "restart"):
        """
        Initialize the database failure test.
        
        Args:
            namespace: Kubernetes namespace where the system is deployed
            recovery_timeout: Maximum time in seconds to wait for recovery
            database_service: Database service name ("postgres" or "redis")
            failure_duration: Duration of the failure in seconds
            failure_type: Type of failure to simulate ("restart", "corrupt", "lock")
        """
        super().__init__(namespace, recovery_timeout)
        self.database_service = database_service
        self.failure_duration = failure_duration
        self.failure_type = failure_type
        self.target_pods = []
        self.original_state = {}
        
    def setup(self):
        """Set up the test by identifying database pods."""
        # Get all pods in the namespace
        pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
        
        # Find database pods
        for pod in pods.items:
            if pod.status.phase == "Running":
                # Match by label or name
                if (
                    self.database_service in pod.metadata.name or 
                    pod.metadata.labels.get("app") == self.database_service
                ):
                    self.target_pods.append(pod)
        
        # Ensure we have pods to target
        if not self.target_pods:
            raise ValueError(f"No pods found for database service: {self.database_service}")
            
        logger.info(f"Found {len(self.target_pods)} {self.database_service} pods for failure testing")
        
    def execute(self):
        """Execute the database failure based on the failure type."""
        for pod in self.target_pods:
            try:
                pod_name = pod.metadata.name
                logger.info(f"Simulating {self.failure_type} failure on {self.database_service} pod {pod_name}")
                
                if self.failure_type == "restart":
                    # Delete the pod to force a restart
                    logger.info(f"Deleting {self.database_service} pod to simulate restart")
                    self.k8s_core_v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=self.namespace,
                        body={}
                    )
                    
                elif self.failure_type == "corrupt":
                    # Simulate data corruption
                    if self.database_service == "postgres":
                        # For Postgres - corrupt a system table
                        exec_command = [
                            "/bin/sh",
                            "-c",
                            "pg_ctl -D /var/lib/postgresql/data stop -m immediate && " +
                            "sleep 2 && " +
                            "echo 'Corrupting data file...' && " +
                            "dd if=/dev/urandom of=/var/lib/postgresql/data/global/pg_control bs=8192 count=1 conv=notrunc && " +
                            "sleep 2 && " +
                            "pg_ctl -D /var/lib/postgresql/data start"
                        ]
                    else:  # redis
                        # For Redis - corrupt the RDB file
                        exec_command = [
                            "/bin/sh",
                            "-c",
                            "redis-cli SAVE && " +
                            "sleep 2 && " +
                            "echo 'Corrupting Redis dump.rdb...' && " +
                            "dd if=/dev/urandom of=/data/dump.rdb seek=10 bs=1 count=10 conv=notrunc && " +
                            "redis-cli SHUTDOWN NOSAVE && " +
                            "sleep 2"
                        ]
                        
                    self.k8s_core_v1.connect_get_namespaced_pod_exec(
                        name=pod_name,
                        namespace=self.namespace,
                        command=exec_command,
                        stderr=True,
                        stdin=False,
                        stdout=True,
                        tty=False
                    )
                    
                elif self.failure_type == "lock":
                    # Simulate database locks/blocking
                    if self.database_service == "postgres":
                        # For Postgres - create a transaction that holds locks
                        exec_command = [
                            "/bin/sh",
                            "-c",
                            "PGPASSWORD=$POSTGRES_PASSWORD psql -U postgres -d aitradingagent -c \"" +
                            "BEGIN; " +
                            "LOCK TABLE users IN ACCESS EXCLUSIVE MODE; " +
                            "SELECT pg_sleep(60); " +
                            "COMMIT;\" &"
                        ]
                    else:  # redis
                        # For Redis - simulate high load and blocked commands
                        exec_command = [
                            "/bin/sh",
                            "-c",
                            "redis-cli MULTI && " +
                            "redis-cli SET stress_key large_value_to_block_redis && " +
                            "for i in {1..1000}; do " +
                            "  redis-cli SADD stress_set $i & " +
                            "done && " +
                            "redis-cli EXEC && " +
                            "sleep 10 && " +
                            "redis-cli FLUSHALL"
                        ]
                        
                    self.k8s_core_v1.connect_get_namespaced_pod_exec(
                        name=pod_name,
                        namespace=self.namespace,
                        command=exec_command,
                        stderr=True,
                        stdin=False,
                        stdout=True,
                        tty=False
                    )
                
                # Wait during failure period
                logger.info(f"Database failure initiated, waiting for {self.failure_duration} seconds")
                time.sleep(self.failure_duration)
                
            except ApiException as e:
                logger.error(f"Error executing database failure on pod {pod.metadata.name}: {e}")
    
    def verify(self) -> bool:
        """
        Verify the system recovered correctly after database failures.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        # Allow additional time for recovery
        recovery_wait = 60
        logger.info(f"Waiting {recovery_wait} seconds for database and dependent services to recover")
        time.sleep(recovery_wait)
        
        # Check if database pod is running
        try:
            pods = self.k8s_core_v1.list_namespaced_pod(namespace=self.namespace)
            db_pods_running = 0
            
            for pod in pods.items:
                if (
                    self.database_service in pod.metadata.name or 
                    pod.metadata.labels.get("app") == self.database_service
                ):
                    if pod.status.phase == "Running":
                        db_pods_running += 1
            
            if db_pods_running == 0:
                logger.error(f"No {self.database_service} pods running after failure test")
                return False
                
            logger.info(f"{db_pods_running} {self.database_service} pods running after recovery")
            
            # Check that dependent services are also running
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
            for deployment in deployments.items:
                if not (self.database_service in deployment.metadata.name):
                    if deployment.status.available_replicas != deployment.status.replicas:
                        logger.error(
                            f"Deployment {deployment.metadata.name} has only "
                            f"{deployment.status.available_replicas}/{deployment.status.replicas} "
                            f"available replicas after database failure"
                        )
                        return False
            
            # Check database connectivity
            if self.database_service == "postgres":
                # Test PostgreSQL connectivity
                for pod in pods.items:
                    if pod.metadata.name.startswith("api"):
                        exec_command = [
                            "/bin/sh",
                            "-c",
                            "python -c \"import os, psycopg2; "
                            "conn = psycopg2.connect(os.environ.get('DATABASE_URL')); "
                            "cur = conn.cursor(); "
                            "cur.execute('SELECT 1'); "
                            "print(cur.fetchone()); "
                            "conn.close()\""
                        ]
                        
                        try:
                            resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                                name=pod.metadata.name,
                                namespace=self.namespace,
                                command=exec_command,
                                stderr=True,
                                stdin=False,
                                stdout=True,
                                tty=False
                            )
                            
                            if "(1,)" not in resp:
                                logger.error(f"PostgreSQL connectivity check failed: {resp}")
                                return False
                                
                            logger.info("PostgreSQL connectivity verified")
                            break
                        except:
                            continue
                            
            else:  # redis
                # Test Redis connectivity
                for pod in pods.items:
                    if pod.metadata.name.startswith("api"):
                        exec_command = [
                            "/bin/sh",
                            "-c",
                            "python -c \"import os, redis; "
                            "r = redis.from_url(os.environ.get('REDIS_URL')); "
                            "r.set('test_key', 'test_value'); "
                            "print(r.get('test_key').decode('utf-8'))\""
                        ]
                        
                        try:
                            resp = self.k8s_core_v1.connect_get_namespaced_pod_exec(
                                name=pod.metadata.name,
                                namespace=self.namespace,
                                command=exec_command,
                                stderr=True,
                                stdin=False,
                                stdout=True,
                                tty=False
                            )
                            
                            if "test_value" not in resp:
                                logger.error(f"Redis connectivity check failed: {resp}")
                                return False
                                
                            logger.info("Redis connectivity verified")
                            break
                        except:
                            continue
            
            # If we made it here, verification passed
            return True
            
        except ApiException as e:
            logger.error(f"Error during verification: {e}")
            return False
    
    def cleanup(self):
        """Clean up after the test."""
        # For most failure types, cleanup happens automatically through Kubernetes
        # For specific manual cleanup:
        if self.failure_type == "lock" and self.database_service == "postgres":
            try:
                # Terminate any lingering transactions
                for pod in self.target_pods:
                    exec_command = [
                        "/bin/sh",
                        "-c",
                        "PGPASSWORD=$POSTGRES_PASSWORD psql -U postgres -d aitradingagent -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction';\""
                    ]
                    
                    self.k8s_core_v1.connect_get_namespaced_pod_exec(
                        name=pod.metadata.name,
                        namespace=self.namespace,
                        command=exec_command,
                        stderr=True,
                        stdin=False,
                        stdout=True,
                        tty=False
                    )
                    
            except ApiException as e:
                logger.error(f"Error during cleanup: {e}")

def test_postgres_restart():
    """Run the PostgreSQL restart test as a pytest function."""
    test = DatabaseFailureTest(
        database_service="postgres",
        failure_type="restart",
        failure_duration=60
    )
    assert test.run() is True

def test_redis_restart():
    """Run the Redis restart test as a pytest function."""
    test = DatabaseFailureTest(
        database_service="redis",
        failure_type="restart",
        failure_duration=60
    )
    assert test.run() is True

if __name__ == "__main__":
    # Run directly
    test = DatabaseFailureTest(database_service="postgres")
    test.run()

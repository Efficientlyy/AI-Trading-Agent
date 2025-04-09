"""
Simple in-memory background job queue for backtests and long tasks.
"""

import asyncio
from typing import Callable, Any, Dict, List

class JobQueue:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}

    async def add_job(self, job_id: str, func: Callable, *args, **kwargs):
        job = {"status": "queued", "result": None}
        self.jobs[job_id] = job
        await self.queue.put((job_id, func, args, kwargs))

    async def worker(self):
        while True:
            job_id, func, args, kwargs = await self.queue.get()
            job = self.jobs[job_id]
            job["status"] = "running"
            try:
                result = await func(*args, **kwargs)
                job["result"] = result
                job["status"] = "completed"
            except Exception as e:
                job["result"] = str(e)
                job["status"] = "failed"

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id, {"status": "not found"})

job_queue = JobQueue()
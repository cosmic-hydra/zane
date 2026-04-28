from collections.abc import Callable
from typing import Any


class MockBacalhauClient:
    def submit(self, spec: dict) -> str:
        print(f"Mock Bacalhau submit: {spec}")
        return "mock_job_123"

    def wait(self, job_id: str) -> dict:
        print(f"Mock Bacalhau wait for {job_id}")
        return {"status": "success", "results": ["/ipfs/mock_crispr_data.json"], "outputs": {"data": "optimized results"}}

class OSKernel:
    def __init__(self):
        self.zkp_callback: Callable[[dict], None] = None

    def set_zkp_callback(self, callback: Callable[[dict], None]):
        self.zkp_callback = callback

    def compile_labop(self, protocol_str: str) -> dict[str, Any]:
        "Compile LabOP protocol to hardware-agnostic executable"
        try:
            # LabOP integration stub
            from labop.core import Protocol  # assume pip install labop
            protocol = Protocol.from_string(protocol_str)
            compiler = protocol.compiler()
            job_spec = compiler.compile()
            return job_spec.as_dict()
        except ImportError:
            print("LabOP not available, mock compilation")
            return {
                "engine": "docker",
                "spec": {
                    "run": protocol_str,
                    "inputs": {}
                }
            }

    def dispatch_bacalhau(self, job_spec: dict[str, Any]) -> dict[str, Any]:
        "Dispatch job to Bacalhau for distributed execution"
        try:
            from bacalhau.apiclient import api_client
            client = api_client.Client()
            with client.new_request_ctx():
                job_request = client.submit(job_spec)
                result = job_request.wait()
                return result.to_dict()
        except ImportError:
            print("Bacalhau SDK not available, mock dispatch")
            mock_client = MockBacalhauClient()
            return mock_client.wait(mock_client.submit(job_spec))

    def ingest_results(self, results: dict[str, Any]):
        "Closed-loop data ingestion with ZKP callback"
        print(f"Ingesting results: {results}")
        if self.zkp_callback:
            self.zkp_callback(results)

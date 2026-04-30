import socket
import requests
import logging
import sys
from typing import List

logger = logging.getLogger(__name__)

class CriticalSecurityException(Exception):
    """Raised when the system detects an unauthorized outbound connection."""
    pass

class AirGapEnforcer:
    """
    Enforces a zero-leakage policy for Pharma B2B environments.
    Prevents any IP or proprietary molecular data from leaving the internal network.
    """

    def __init__(self, allowed_hosts: List[str] = None):
        self.allowed_hosts = allowed_hosts or ["localhost", "127.0.0.1"]
        self._original_socket = socket.socket

    def enforce_strict_isolation(self):
        """
        Overrides the global socket object to block unauthorized outbound traffic.
        This effectively kills any accidental calls to OpenAI, WandB, or Telemetry.
        """
        def guarded_connect(instance, address):
            host = address[0]
            if host not in self.allowed_hosts:
                logger.critical(f"UNAUTHORIZED OUTBOUND ATTEMPT: {host}")
                raise CriticalSecurityException(
                    f"Outbound connection to {host} blocked by AirGapEnforcer. "
                    "Proprietary IP protection active."
                )
            return self._original_socket.connect(instance, address)

        # Monkey-patch socket to prevent low-level networking
        socket.socket.connect = guarded_connect
        logger.info("Air-gap network isolation active. All outbound traffic intercepted.")

    def verify_offline_mode(self):
        """
        Pings known external servers (e.g., Google DNS, PyPI).
        If connection succeeds, the environment is NOT air-gapped.
        """
        test_targets = ["8.8.8.8", "pypi.org", "google.com"]
        
        for target in test_targets:
            try:
                # Use a very short timeout
                requests.get(f"http://{target}", timeout=1.0)
                
                # If we reach here, we are NOT air-gapped
                logger.error(f"SECURITY BREACH: System can reach {target}. Environment is not isolated.")
                raise CriticalSecurityException(
                    "Internet detected. ZANE refuses to boot models in an insecure environment. "
                    "Ensure the host is air-gapped before proceeding."
                )
            except (requests.ConnectionError, requests.Timeout):
                # This is the expected state in a secure pharma lab
                logger.info(f"Verified isolation from {target}")
                continue
            except CriticalSecurityException:
                raise
            except Exception as e:
                # Catch-all for other network errors which also imply isolation
                logger.info(f"Connection to {target} failed as expected: {e}")

        logger.info("Environment isolation verified. Safe for proprietary data processing.")

def initialize_security():
    enforcer = AirGapEnforcer()
    try:
        enforcer.verify_offline_mode()
        enforcer.enforce_strict_isolation()
    except CriticalSecurityException as e:
        print(f"FATAL SECURITY ERROR: {e}")
        sys.exit(1)

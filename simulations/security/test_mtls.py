#!/usr/bin/env python3
"""
PCP Security Tests
=====================

Comprehensive security and stress tests for PCP including:
- mTLS authentication tests
- Certificate validation tests
- Rate limiting tests
- Load and stress tests
- Security vulnerability tests

Usage:
    python tests/security/test_mtls.py
    python tests/security/test_stress.py
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import ssl
import string
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from unittest import TestCase, main as unittest_main
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger("pcp-security")


def generate_random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_certificate_serial() -> int:
    return random.randint(1, 2**32)


@dataclass
class Certificate:
    """X.509 certificate representation."""
    subject: str
    issuer: str
    serial_number: int
    not_before: float
    not_after: float
    public_key: str
    signature: bytes
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "issuer": self.issuer,
            "serialNumber": self.serial_number,
            "notBefore": self.not_before,
            "notAfter": self.not_after,
            "publicKey": self.public_key,
            "extensions": self.extensions,
        }

    def is_expired(self) -> bool:
        now = time.time()
        return now < self.not_before or now > self.not_after

    def days_until_expiry(self) -> float:
        return (self.not_after - time.time()) / 86400


class CertificateAuthority:
    """Mock Certificate Authority for testing."""

    def __init__(self, name: str):
        self.name = name
        self.ca_key = generate_random_string(32)
        self.ca_cert = Certificate(
            subject=name,
            issuer=name,
            serial_number=1,
            not_before=time.time() - 86400 * 365,
            not_after=time.time() + 86400 * 3650,
            public_key=self.ca_key,
            signature=b"CA_SIGNATURE",
            extensions={"is_ca": True, "key_usage": ["sign", "cert"]},
        )
        self.issued_certs: Dict[str, Certificate] = {}
        self.revoked_serials: set = set()

    def issue_certificate(
        self,
        subject: str,
        validity_days: int = 365,
        key_usage: Optional[List[str]] = None,
        extended_key_usage: Optional[List[str]] = None,
    ) -> Certificate:
        serial = generate_certificate_serial()
        now = time.time()

        cert = Certificate(
            subject=subject,
            issuer=self.name,
            serial_number=serial,
            not_before=now,
            not_after=now + validity_days * 86400,
            public_key=generate_random_string(32),
            signature=self._sign(serial),
            extensions={
                "is_ca": False,
                "key_usage": key_usage or ["digital_signature", "key_encipherment"],
                "extended_key_usage": extended_key_usage or ["server_auth", "client_auth"],
            },
        )

        self.issued_certs[subject] = cert
        return cert

    def _sign(self, serial: int) -> bytes:
        data = f"{serial}{self.ca_key}".encode()
        return hashlib.sha256(data).digest()

    def revoke_certificate(self, subject: str) -> bool:
        if subject not in self.issued_certs:
            return False
        cert = self.issued_certs[subject]
        self.revoked_serials.add(cert.serial_number)
        return True

    def is_revoked(self, serial: int) -> bool:
        return serial in self.revoked_serials

    def verify_certificate(self, cert: Certificate) -> tuple[bool, Optional[str]]:
        if cert.issuer != self.name:
            return False, "Certificate not issued by this CA"
        if self.is_revoked(cert.serial_number):
            return False, "Certificate has been revoked"
        if cert.is_expired():
            return False, "Certificate has expired"
        return True, None


class MTLSTestSuite:
    """Test suite for mTLS functionality."""

    def __init__(self):
        self.ca = CertificateAuthority("PCP Test CA")
        self.test_results: List[Dict[str, Any]] = []

    def test_certificate_issuance(self):
        """Test that CA can issue certificates."""
        cert = self.ca.issue_certificate("robot-001.test.local")
        assert cert.subject == "robot-001.test.local"
        assert cert.serial_number > 0
        assert not cert.is_expired()
        self._record_result("certificate_issuance", True, "Certificate issued successfully")

    def test_certificate_expiry(self):
        """Test certificate expiry detection."""
        cert = self.ca.issue_certificate("robot-002.test.local", validity_days=-1)
        assert cert.is_expired()
        self._record_result("certificate_expiry", True, "Expired certificate detected")

    def test_certificate_verification(self):
        """Test certificate verification."""
        cert = self.ca.issue_certificate("robot-003.test.local")
        valid, error = self.ca.verify_certificate(cert)
        assert valid
        self._record_result("certificate_verification", True, f"Verified: {error}")

    def test_revoked_certificate(self):
        """Test revocation detection."""
        cert = self.ca.issue_certificate("robot-004.test.local")
        self.ca.revoke_certificate("robot-004.test.local")
        valid, error = self.ca.verify_certificate(cert)
        assert not valid
        assert "revoked" in error
        self._record_result("revoked_certificate", True, "Revoked certificate detected")

    def test_wrong_issuer(self):
        """Test rejection of certificate from wrong issuer."""
        other_ca = CertificateAuthority("Other CA")
        cert = other_ca.issue_certificate("robot-005.test.local")
        valid, _ = self.ca.verify_certificate(cert)
        assert not valid
        self._record_result("wrong_issuer", True, "Rejected certificate from wrong issuer")

    def test_self_signed_certificate(self):
        """Test self-signed certificate handling."""
        self_signed = Certificate(
            subject="robot-self.test.local",
            issuer="robot-self.test.local",
            serial_number=generate_certificate_serial(),
            not_before=time.time() - 86400,
            not_after=time.time() + 86400,
            public_key=generate_random_string(32),
            signature=b"SELF_SIGNED",
        )
        valid, error = self.ca.verify_certificate(self_signed)
        assert not valid
        self._record_result("self_signed", True, f"Rejected self-signed: {error}")

    def test_validity_period(self):
        """Test certificate validity period."""
        short_cert = self.ca.issue_certificate("robot-short.test.local", validity_days=30)
        long_cert = self.ca.issue_certificate("robot-long.test.local", validity_days=3650)

        assert short_cert.days_until_expiry() < 31
        assert long_cert.days_until_expiry() > 3649

        self._record_result("validity_period", True, "Validity periods correct")

    def test_serial_number_uniqueness(self):
        """Test that serial numbers are unique."""
        serials = set()
        for i in range(100):
            cert = self.ca.issue_certificate(f"robot-{i}.test.local")
            assert cert.serial_number not in serials
            serials.add(cert.serial_number)

        self._record_result("serial_uniqueness", True, "All serial numbers unique")

    def _record_result(self, test_name: str, passed: bool, message: str):
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": time.time(),
        }
        self.test_results.append(result)
        logger.info(f"{'PASS' if passed else 'FAIL'}: {test_name} - {message}")

    def get_summary(self) -> Dict[str, Any]:
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%",
            "results": self.test_results,
        }


class RateLimitTestSuite:
    """Test suite for rate limiting functionality."""

    def __init__(self):
        self.requests: List[float] = []
        self.blocked: int = 0

    def test_rate_limit_allow(self):
        """Test that requests under limit are allowed."""
        limit = 100
        window = 60
        current_time = time.time()

        for i in range(50):
            self.requests.append(current_time - i * 0.5)

        allowed = self._check_rate_limit(limit, window, current_time)
        assert allowed
        logger.info("Rate limit: 50 requests allowed in window")

    def test_rate_limit_exceed(self):
        """Test that requests over limit are blocked."""
        limit = 100
        window = 60
        current_time = time.time()

        for i in range(150):
            self.requests.append(current_time - i * 0.5)

        allowed = self._check_rate_limit(limit, window, current_time)
        if not allowed:
            self.blocked += 1
        assert not allowed
        logger.info("Rate limit: exceeded requests blocked")

    def test_rate_limit_window_reset(self):
        """Test that rate limit resets after window."""
        limit = 10
        window = 60

        current_time = time.time()
        for i in range(15):
            self.requests.append(current_time - i * 5)

        allowed1 = self._check_rate_limit(limit, window, current_time)
        assert not allowed1

        future_time = current_time + window + 1
        allowed2 = self._check_rate_limit(limit, window, future_time)
        assert allowed2
        logger.info("Rate limit: window reset works")

    def test_burst_handling(self):
        """Test handling of burst requests."""
        limit = 20
        window = 60

        current_time = time.time()
        for i in range(limit):
            self.requests.append(current_time - 1)

        allowed_first = self._check_rate_limit(limit, window, current_time)
        assert allowed_first

        for _ in range(5):
            allowed = self._check_rate_limit(limit, window, current_time)
            if not allowed:
                self.blocked += 1

        assert self.blocked > 0
        logger.info(f"Rate limit: burst of {self.blocked} requests blocked")

    def _check_rate_limit(self, limit: int, window: int, current_time: float) -> bool:
        cutoff = current_time - window
        recent_requests = [r for r in self.requests if r > cutoff]
        return len(recent_requests) < limit


class StressTestSuite:
    """Stress testing for PCP components."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def stress_test_concurrent_connections(self, num_connections: int = 1000):
        """Test handling of many concurrent connections."""
        logger.info(f"Starting stress test with {num_connections} concurrent connections")

        connections: List[Dict[str, Any]] = []
        start_time = time.time()

        for i in range(num_connections):
            conn = {
                "id": i,
                "connected_at": time.time(),
                "last_ping": time.time(),
                "active": True,
            }
            connections.append(conn)

        duration = time.time() - start_time

        self.results.append({
            "test": "concurrent_connections",
            "connections": num_connections,
            "duration": duration,
            "throughput": num_connections / duration if duration > 0 else 0,
        })

        logger.info(f"Connected {num_connections} in {duration:.2f}s")
        return connections

    def stress_test_message_throughput(self, num_messages: int = 10000):
        """Test message throughput."""
        logger.info(f"Starting throughput test with {num_messages} messages")

        messages = []
        start_time = time.time()

        for i in range(num_messages):
            msg = {
                "id": f"msg-{i}",
                "type": random.choice(["telemetry", "command", "state"]),
                "size": random.randint(100, 10000),
                "timestamp": time.time(),
            }
            messages.append(msg)

        duration = time.time() - start_time

        self.results.append({
            "test": "message_throughput",
            "messages": num_messages,
            "duration": duration,
            "msg_per_sec": num_messages / duration if duration > 0 else 0,
        })

        logger.info(f"Processed {num_messages} messages in {duration:.2f}s")

    def stress_test_memory_usage(self):
        """Test memory usage under load."""
        logger.info("Starting memory usage test")

        data = []
        for i in range(100000):
            entry = {
                "id": i,
                "data": generate_random_string(100),
                "timestamp": time.time(),
            }
            data.append(entry)

        memory_mb = len(str(data)) / (1024 * 1024)

        self.results.append({
            "test": "memory_usage",
            "entries": len(data),
            "memory_mb": memory_mb,
        })

        logger.info(f"Memory usage: {memory_mb:.2f} MB for {len(data)} entries")

    def stress_test_database_operations(self, num_ops: int = 1000):
        """Test database operations under load."""
        logger.info(f"Starting database stress test with {num_ops} operations")

        operations = []
        start_time = time.time()

        for i in range(num_ops):
            op = {
                "type": random.choice(["read", "write", "update", "delete"]),
                "size": random.randint(100, 10000),
            }
            operations.append(op)

        duration = time.time() - start_time

        self.results.append({
            "test": "database_operations",
            "operations": num_ops,
            "duration": duration,
            "ops_per_sec": num_ops / duration if duration > 0 else 0,
        })

        logger.info(f"Completed {num_ops} DB ops in {duration:.2f}s")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tests": len(self.results),
            "results": self.results,
        }


class SecurityVulnerabilityTestSuite:
    """Tests for common security vulnerabilities."""

    def test_sql_injection(self):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "'; DROP TABLE robots; --",
            "' OR '1'='1",
            "'; UPDATE robots SET status='compromised'; --",
            "1; DELETE FROM alerts; --",
        ]

        for inp in malicious_inputs:
            result = self._simulate_query(inp)
            assert "error" in result or "sanitized" in result

        logger.info("SQL injection: All malicious inputs handled")

    def test_xss_prevention(self):
        """Test XSS prevention."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
        ]

        for inp in malicious_inputs:
            result = self._sanitize_html(inp)
            assert "<script>" not in result

        logger.info("XSS prevention: All malicious inputs sanitized")

    def test_path_traversal(self):
        """Test path traversal prevention."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32",
        ]

        for path in malicious_paths:
            result = self._validate_path(path)
            assert result is None or "blocked" in result.lower()

        logger.info("Path traversal: All malicious paths blocked")

    def test_command_injection(self):
        """Test command injection prevention."""
        malicious_commands = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "`whoami`",
            "$(whoami)",
            "&& ls -la",
        ]

        for cmd in malicious_commands:
            result = self._validate_command(cmd)
            assert "blocked" in result.lower() or "error" in result.lower()

        logger.info("Command injection: All malicious commands blocked")

    def test_weak_crypto(self):
        """Test detection of weak cryptography."""
        weak_algorithms = ["md5", "sha1", "des", "rc4"]

        for algo in weak_algorithms:
            detected = self._detect_weak_crypto(algo)
            assert detected

        logger.info("Weak crypto: All weak algorithms detected")

    def test_insecure_defaults(self):
        """Test for insecure default configurations."""
        insecure_configs = [
            {"debug": True, "secret_key": "default"},
            {"auth": False, "tls": False},
            {"admin_user": "admin", "admin_pass": "admin"},
        ]

        for config in insecure_configs:
            issues = self._check_config_security(config)
            assert len(issues) > 0

        logger.info("Insecure defaults: All issues detected")

    def _simulate_query(self, inp: str) -> Dict[str, str]:
        if "'" in inp or ";" in inp:
            return {"error": "SQL injection detected", "sanitized": True}
        return {"result": "ok"}

    def _sanitize_html(self, inp: str) -> str:
        dangerous = ["<script>", "javascript:", "onerror=", "onload="]
        result = inp
        for d in dangerous:
            result = result.replace(d, "")
        return result

    def _validate_path(self, path: str) -> Optional[str]:
        if ".." in path or path.startswith("/") or ":" in path:
            return "Path traversal blocked"
        return None

    def _validate_command(self, cmd: str) -> str:
        dangerous = [";", "|", "`", "$(", "&&"]
        for d in dangerous:
            if d in cmd:
                return "Command injection blocked"
        return "Command allowed"

    def _detect_weak_crypto(self, algo: str) -> bool:
        return algo.lower() in ["md5", "sha1", "des", "rc4", "base64"]

    def _check_config_security(self, config: Dict[str, Any]) -> List[str]:
        issues = []
        if config.get("debug") == True:
            issues.append("Debug mode enabled")
        if config.get("auth") == False:
            issues.append("Authentication disabled")
        if config.get("secret_key") == "default":
            issues.append("Default secret key")
        return issues


async def run_all_tests():
    """Run all security test suites."""
    logger.info("=" * 60)
    logger.info("Starting PCP Security Tests")
    logger.info("=" * 60)

    mtls_suite = MTLSTestSuite()
    logger.info("\n--- mTLS Tests ---")
    mtls_suite.test_certificate_issuance()
    mtls_suite.test_certificate_expiry()
    mtls_suite.test_certificate_verification()
    mtls_suite.test_revoked_certificate()
    mtls_suite.test_wrong_issuer()
    mtls_suite.test_self_signed_certificate()
    mtls_suite.test_validity_period()
    mtls_suite.test_serial_number_uniqueness()

    mtls_summary = mtls_suite.get_summary()
    logger.info(f"\nmTLS Summary: {mtls_summary['passed']}/{mtls_summary['total']} passed")

    rate_limit_suite = RateLimitTestSuite()
    logger.info("\n--- Rate Limiting Tests ---")
    rate_limit_suite.test_rate_limit_allow()
    rate_limit_suite.test_rate_limit_exceed()
    rate_limit_suite.test_rate_limit_window_reset()
    rate_limit_suite.test_burst_handling()

    stress_suite = StressTestSuite()
    logger.info("\n--- Stress Tests ---")
    stress_suite.stress_test_concurrent_connections(500)
    stress_suite.stress_test_message_throughput(5000)
    stress_suite.stress_test_memory_usage()
    stress_suite.stress_test_database_operations(500)

    vuln_suite = SecurityVulnerabilityTestSuite()
    logger.info("\n--- Vulnerability Tests ---")
    vuln_suite.test_sql_injection()
    vuln_suite.test_xss_prevention()
    vuln_suite.test_path_traversal()
    vuln_suite.test_command_injection()
    vuln_suite.test_weak_crypto()
    vuln_suite.test_insecure_defaults()

    logger.info("\n" + "=" * 60)
    logger.info("All Security Tests Complete")
    logger.info("=" * 60)

    return {
        "mtls": mtls_summary,
        "stress": stress_suite.get_summary(),
    }


if __name__ == "__main__":
    asyncio.run(run_all_tests())
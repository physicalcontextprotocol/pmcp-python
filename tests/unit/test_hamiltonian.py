"""
Unit tests for Phase 1 — Hamiltonian Neural Network core.
Tests: HNN forward, encoder, integrators, conservation checker.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import pytest

np = pytest.importorskip(
    "numpy", reason="numpy not installed — install the 'numerics' or 'hnn' extra"
)
torch = pytest.importorskip(
    "torch", reason="torch not installed — install the 'hnn' extra (pip install pcp[hnn])"
)

from pcp.physics.hamiltonian import (
    HamiltonianNN,
    PhaseSpaceEncoder,
    StormerVerlet,
    ConservationChecker,
)


class TestHamiltonianNN:

    def test_init_default(self):
        hnn = HamiltonianNN(q_dim=6, p_dim=6)
        assert hnn.q_dim == 6
        assert hnn.p_dim == 6
        assert hnn.input_dim == 12

    def test_forward_shape(self):
        hnn = HamiltonianNN(q_dim=6, p_dim=6)
        q = torch.randn(6)
        p = torch.randn(6)
        H = hnn.forward(q, p)
        assert H.shape == ()

    def test_forward_batch(self):
        hnn = HamiltonianNN(q_dim=6, p_dim=6)
        q = torch.randn(4, 6)
        p = torch.randn(4, 6)
        H = hnn.forward(q, p)
        assert H.shape == (4,)

    def test_hamiltonian_numpy(self):
        hnn = HamiltonianNN(q_dim=6, p_dim=6)
        q = np.random.randn(6).astype(np.float32)
        p = np.random.randn(6).astype(np.float32)
        H = hnn.hamiltonian(q, p)
        assert isinstance(H, float)
        assert not np.isnan(H)

    def test_hamiltonians_batch(self):
        hnn = HamiltonianNN(q_dim=6, p_dim=6)
        q = np.random.randn(10, 6).astype(np.float32)
        p = np.random.randn(10, 6).astype(np.float32)
        H = hnn.hamiltonians(q, p)
        assert H.shape == (10,)

    def test_integrate_symplectic(self):
        hnn = HamiltonianNN(q_dim=6, p_dim=6)
        q0 = np.zeros(6, dtype=np.float32)
        p0 = np.ones(6, dtype=np.float32)
        traj_q, traj_p = hnn.integrate_symplectic(q0, p0, steps=10, dt=0.01)
        assert traj_q.shape == (11, 6)
        assert traj_p.shape == (11, 6)
        assert np.all(np.isfinite(traj_q))
        assert np.all(np.isfinite(traj_p))

    def test_compute_dynamics(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        q = torch.randn(4, requires_grad=True)
        p = torch.randn(4, requires_grad=True)
        dq_dt, dp_dt = hnn.compute_dynamics(q, p)
        assert dq_dt.shape == (4,)
        assert dp_dt.shape == (4,)

    def test_train_step(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        q = np.random.randn(32, 4).astype(np.float32)
        p = np.random.randn(32, 4).astype(np.float32)
        loss = hnn.train_step(q, p)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_save_load(self, tmp_path):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        path = tmp_path / "hnn.pt"
        hnn.save(str(path))
        assert path.exists()

        hnn2 = HamiltonianNN(q_dim=4, p_dim=4)
        hnn2.load(str(path))

        q = np.random.randn(4).astype(np.float32)
        p = np.random.randn(4).astype(np.float32)
        H1 = hnn.hamiltonian(q, p)
        H2 = hnn2.hamiltonian(q, p)
        assert abs(H1 - H2) < 1e-5


class TestPhaseSpaceEncoder:

    def test_encode_basic(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        sensor_data = {
            "joint_angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "joint_velocities": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
        q, p = encoder.encode(sensor_data)
        assert len(q) == 6
        assert len(p) == 6

    def test_encode_with_pose(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        sensor_data = {
            "joint_angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "joint_velocities": [0.01] * 6,
            "end_effector_pose": {"x": 0.5, "y": 0.3, "z": 0.2},
        }
        q, p = encoder.encode(sensor_data)
        assert len(q) >= 9

    def test_encode_with_mass_matrix(self):
        encoder = PhaseSpaceEncoder(n_joints=4)
        sensor_data = {
            "joint_angles": [0.1, 0.2, 0.3, 0.4],
            "joint_velocities": [1.0, 1.0, 1.0, 1.0],
            "mass_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        }
        q, p = encoder.encode(sensor_data)
        assert len(p) == 4

    def test_decode(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.3, 0.2])
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.0, 0.0, 0.0])
        decoded = encoder.decode(q, p)
        assert "joint_angles" in decoded
        assert "end_effector_pose" in decoded

    def test_validate_valid(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        q = np.random.randn(6)
        p = np.random.randn(6)
        valid, msg = encoder.validate(q, p)
        assert valid

    def test_validate_dimension_mismatch(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        q = np.random.randn(6)
        p = np.random.randn(4)
        valid, msg = encoder.validate(q, p)
        assert not valid
        assert "mismatch" in msg.lower()

    def test_validate_nan(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        q = np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6])
        p = np.random.randn(6)
        valid, msg = encoder.validate(q, p)
        assert not valid


class TestSymplecticIntegrators:

    def setup_method(self):
        # HamiltonianNN uses unseeded random weight init. Without a fixed
        # seed, energy-drift assertions below are a function of the luck
        # of the draw rather than the integrator's actual behavior, which
        # makes the suite flaky (observed: ~1-in-N runs exceed the drift
        # tolerance purely from initialization variance). Seed both RNGs
        # so these tests are deterministic and reproducible.
        np.random.seed(0)
        torch.manual_seed(0)

    def test_stormer_verlet_step(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        integrator = StormerVerlet(hnn)

        q = np.random.randn(4).astype(np.float32)
        p = np.random.randn(4).astype(np.float32)
        q_new, p_new = integrator.step(q, p, dt=0.01)

        assert q_new.shape == (4,)
        assert p_new.shape == (4,)
        assert np.all(np.isfinite(q_new))
        assert np.all(np.isfinite(p_new))

    def test_stormer_verlet_integrate(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        integrator = StormerVerlet(hnn)

        q0 = np.zeros(4, dtype=np.float32)
        p0 = np.ones(4, dtype=np.float32)
        traj_q, traj_p = integrator.integrate(q0, p0, steps=5, dt=0.01)

        assert traj_q.shape == (6, 4)
        assert traj_p.shape == (6, 4)

    def test_energy_conservation_over_time(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        q0 = np.zeros(4, dtype=np.float32)
        p0 = np.ones(4, dtype=np.float32) * 0.1

        integrator = StormerVerlet(hnn)
        traj_q, traj_p = integrator.integrate(q0, p0, steps=100, dt=0.01)

        energies = [hnn.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
        energies = np.array(energies)

        H0 = energies[0]
        H_final = energies[-1]
        relative_drift = abs(H_final - H0) / abs(H0) if abs(H0) > 1e-8 else 0.0

        assert relative_drift < 0.5, f"Energy drift too large: {relative_drift}"


class TestConservationChecker:

    def test_init(self):
        checker = ConservationChecker(threshold=0.05)
        assert checker.threshold == 0.05
        assert checker.relative is True

    def test_drift_from_arrays(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        checker = ConservationChecker(threshold=0.05)

        q0 = np.zeros(4, dtype=np.float32)
        p0 = np.ones(4, dtype=np.float32) * 0.1
        traj_q, traj_p = hnn.integrate_symplectic(q0, p0, steps=50, dt=0.01)

        drift = checker.drift_from_arrays(traj_q, traj_p, hnn)
        assert isinstance(drift, float)
        assert drift >= 0

    def test_check_violation(self):
        checker = ConservationChecker(threshold=0.05)
        is_violation, reason = checker.check_violation(0.03)
        assert not is_violation

        is_violation, reason = checker.check_violation(0.10)
        assert is_violation
        assert "exceeds" in reason.lower()

    def test_analyze_trajectory(self):
        hnn = HamiltonianNN(q_dim=4, p_dim=4)
        checker = ConservationChecker(threshold=0.05)

        q0 = np.zeros(4, dtype=np.float32)
        p0 = np.ones(4, dtype=np.float32) * 0.1
        traj_q, traj_p = hnn.integrate_symplectic(q0, p0, steps=50, dt=0.01)

        analysis = checker.analyze_trajectory(traj_q, traj_p, hnn)
        assert "H0" in analysis
        assert "H_final" in analysis
        assert "drift" in analysis
        assert "is_violation" in analysis

    def test_get_safety_status(self):
        checker = ConservationChecker(threshold=0.05)
        assert checker.get_safety_status(0.01) == "NOMINAL"
        assert checker.get_safety_status(0.03) == "CAUTION"
        assert checker.get_safety_status(0.07) == "WARNING"
        assert checker.get_safety_status(0.20) == "ALERT"
        assert checker.get_safety_status(0.50) == "CRITICAL"

    def test_severity_levels(self):
        checker = ConservationChecker(threshold=0.05)
        _, r1 = checker.check_violation(0.06)
        assert "minor" in r1

        _, r2 = checker.check_violation(0.20)
        assert "moderate" in r2

        _, r3 = checker.check_violation(0.50)
        assert "major" in r3

        _, r4 = checker.check_violation(1.0)
        assert "critical" in r4


class TestIntegrationFlow:

    def test_encoder_to_hnn_to_integrator(self):
        encoder = PhaseSpaceEncoder(n_joints=4)
        hnn = HamiltonianNN(q_dim=4, p_dim=4)

        sensor_data = {
            "joint_angles": [0.1, 0.2, 0.3, 0.4],
            "joint_velocities": [0.01, 0.02, 0.03, 0.04],
        }

        q, p = encoder.encode(sensor_data)
        H = hnn.hamiltonian(q, p)
        assert isinstance(H, float)

        traj_q, traj_p = hnn.integrate_symplectic(q, p, steps=10, dt=0.01)
        assert traj_q.shape[0] == 11

    def test_full_pipeline(self):
        encoder = PhaseSpaceEncoder(n_joints=6)
        hnn = HamiltonianNN(q_dim=12, p_dim=12)
        checker = ConservationChecker(threshold=0.05)

        sensor_data = {
            "joint_angles": [0.1] * 6,
            "joint_velocities": [0.01] * 6,
            "end_effector_pose": {"x": 0.5, "y": 0.3, "z": 0.2},
        }

        q, p = encoder.encode(sensor_data)
        n = min(len(q), len(p))
        q = np.array(q[:n], dtype=np.float32)
        p = np.array(p[:n], dtype=np.float32)

        if q.shape[0] < 12:
            q = np.pad(q, (0, 12 - q.shape[0]))
            p = np.pad(p, (0, 12 - p.shape[0]))

        H = hnn.hamiltonian(q, p)

        traj_q, traj_p = hnn.integrate_symplectic(q, p, steps=50, dt=0.01)
        drift = checker.drift_from_arrays(traj_q, traj_p, hnn)

        is_violation, reason = checker.check_violation(drift)
        status = checker.get_safety_status(drift)

        assert isinstance(drift, float)
        assert isinstance(status, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""EnergyForge specialized agents (CrewAI) — five roles, scoped tool sets."""

from agents.base import (
    AgentOutputBase,
    AgentRunResult,
    build_llm,
    llm_complete,
    run_agent,
)
from agents.diagnostics import DiagnosisReport, run_diagnostics
from agents.maintenance import MaintenanceRecommendation, run_maintenance
from agents.optimization import OptimizationPlan, run_optimization
from agents.reporting import ReportArtifact, run_reporting
from agents.safety import SafetyAssessment, run_safety

__all__ = [
    "AgentOutputBase",
    "AgentRunResult",
    "DiagnosisReport",
    "MaintenanceRecommendation",
    "OptimizationPlan",
    "ReportArtifact",
    "SafetyAssessment",
    "build_llm",
    "llm_complete",
    "run_agent",
    "run_diagnostics",
    "run_maintenance",
    "run_optimization",
    "run_reporting",
    "run_safety",
]

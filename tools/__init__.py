"""EnergyForge tool library — nine production tools, one contract.

Every tool subclasses :class:`EnergyForgeTool`: Pydantic in/out schemas, a
mandatory ``confidence`` field on outputs, and a never-raise guarantee
(failures arrive via the ``error`` field).
"""

from tools.anomaly_detector import (
    AnomalyDetectionOutput,
    AnomalyDetectorInput,
    AnomalyDetectorTool,
)
from tools.base import (
    BaseToolInput,
    BaseToolOutput,
    EnergyForgeTool,
    ToolCallRecord,
    ToolCallRecorder,
    bind_tool_owner_loop,
)
from tools.document_generator import (
    DocumentGeneratorInput,
    DocumentGeneratorOutput,
    DocumentGeneratorTool,
    ReportSection,
)
from tools.notification import NotificationInput, NotificationOutput, NotificationTool
from tools.optimization import OptimizationInput, OptimizationOutput, OptimizationTool
from tools.prognostics import PrognosticsInput, PrognosticsOutput, PrognosticsTool
from tools.risk_calculator import (
    RiskCalculatorInput,
    RiskCalculatorOutput,
    RiskCalculatorTool,
)
from tools.sensor_query import SensorQueryInput, SensorQueryOutput, SensorQueryTool
from tools.weather import WeatherInput, WeatherOutput, WeatherTool
from tools.work_order import WorkOrderInput, WorkOrderOutput, WorkOrderTool

TOOL_REGISTRY: dict[str, type[EnergyForgeTool]] = {
    cls.name: cls
    for cls in (
        SensorQueryTool,
        AnomalyDetectorTool,
        PrognosticsTool,
        WeatherTool,
        OptimizationTool,
        WorkOrderTool,
        DocumentGeneratorTool,
        RiskCalculatorTool,
        NotificationTool,
    )
}

__all__ = [
    "TOOL_REGISTRY",
    "AnomalyDetectionOutput",
    "AnomalyDetectorInput",
    "AnomalyDetectorTool",
    "BaseToolInput",
    "BaseToolOutput",
    "DocumentGeneratorInput",
    "DocumentGeneratorOutput",
    "DocumentGeneratorTool",
    "EnergyForgeTool",
    "NotificationInput",
    "NotificationOutput",
    "NotificationTool",
    "OptimizationInput",
    "OptimizationOutput",
    "OptimizationTool",
    "PrognosticsInput",
    "PrognosticsOutput",
    "PrognosticsTool",
    "ReportSection",
    "RiskCalculatorInput",
    "RiskCalculatorOutput",
    "RiskCalculatorTool",
    "SensorQueryInput",
    "SensorQueryOutput",
    "SensorQueryTool",
    "ToolCallRecord",
    "ToolCallRecorder",
    "WeatherInput",
    "WeatherOutput",
    "WeatherTool",
    "WorkOrderInput",
    "WorkOrderOutput",
    "WorkOrderTool",
    "bind_tool_owner_loop",
]

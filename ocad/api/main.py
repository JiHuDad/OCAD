"""Main FastAPI application."""

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import configure_logging, get_logger
from ..core.models import Alert, Endpoint, KPIMetrics, Severity
from ..system.orchestrator import SystemOrchestrator


# Configure logging
configure_logging(settings.monitoring.log_level)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ORAN CFM-Lite AI Anomaly Detection",
    description="Capability-driven hybrid anomaly detection for ORAN environments",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[SystemOrchestrator] = None


class EndpointRequest(BaseModel):
    """Request model for adding endpoints."""
    host: str
    port: int = 830
    role: str = "o-ru"


class AlertResponse(BaseModel):
    """Response model for alerts."""
    alerts: List[Alert]
    total: int


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    endpoints: int
    active_alerts: int
    capability_coverage: float
    processing_latency_p95: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global orchestrator
    
    logger.info("Starting OCAD system", version="0.1.0")
    
    try:
        orchestrator = SystemOrchestrator(settings)
        await orchestrator.start()
        logger.info("OCAD system started successfully")
    except Exception as e:
        logger.error("Failed to start OCAD system", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global orchestrator
    
    logger.info("Shutting down OCAD system")
    
    if orchestrator:
        await orchestrator.stop()
    
    logger.info("OCAD system stopped")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "ORAN CFM-Lite AI Anomaly Detection",
        "version": "0.1.0",
        "status": "operational" if orchestrator and orchestrator.is_running else "stopped"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    health_status = await orchestrator.get_health_status()
    
    if not health_status.get("healthy", False):
        raise HTTPException(status_code=503, detail="System unhealthy")
    
    return health_status


@app.get("/endpoints", response_model=List[Endpoint])
async def list_endpoints():
    """List all registered endpoints."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return await orchestrator.list_endpoints()


@app.post("/endpoints")
async def add_endpoint(request: EndpointRequest):
    """Add a new endpoint for monitoring."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        endpoint = Endpoint(
            id=f"{request.host}:{request.port}",
            host=request.host,
            port=request.port,
            role=request.role,
        )
        
        success = await orchestrator.add_endpoint(endpoint)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add endpoint")
        
        return {"message": "Endpoint added successfully", "endpoint_id": endpoint.id}
        
    except Exception as e:
        logger.error("Failed to add endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/endpoints/{endpoint_id}")
async def remove_endpoint(endpoint_id: str):
    """Remove an endpoint from monitoring."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = await orchestrator.remove_endpoint(endpoint_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    return {"message": "Endpoint removed successfully"}


@app.get("/alerts", response_model=AlertResponse)
async def list_alerts(
    severity: Optional[Severity] = Query(None, description="Filter by severity"),
    endpoint_id: Optional[str] = Query(None, description="Filter by endpoint"),
    limit: int = Query(100, description="Maximum number of alerts"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """List active alerts with optional filtering."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    alerts = await orchestrator.list_alerts(
        severity=severity,
        endpoint_id=endpoint_id,
        limit=limit,
        offset=offset,
    )
    
    return AlertResponse(
        alerts=alerts,
        total=len(alerts),
    )


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = await orchestrator.acknowledge_alert(alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"message": "Alert acknowledged successfully"}


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = await orchestrator.resolve_alert(alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"message": "Alert resolved successfully"}


@app.post("/endpoints/{endpoint_id}/suppress")
async def suppress_endpoint(endpoint_id: str, duration_minutes: int = 60):
    """Suppress alerts for an endpoint."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    await orchestrator.suppress_endpoint(endpoint_id, duration_minutes * 60)
    
    return {"message": f"Endpoint suppressed for {duration_minutes} minutes"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = await orchestrator.get_statistics()
    
    return StatsResponse(
        endpoints=stats.get("endpoints", 0),
        active_alerts=stats.get("active_alerts", 0),
        capability_coverage=stats.get("capability_coverage", 0.0),
        processing_latency_p95=stats.get("processing_latency_p95"),
    )


@app.get("/kpi")
async def get_kpi_metrics():
    """Get KPI metrics for the system."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    kpi_metrics = await orchestrator.get_kpi_metrics()
    
    return kpi_metrics


@app.get("/debug/windows")
async def get_window_stats():
    """Get debug information about feature windows."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return await orchestrator.get_debug_info()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ocad.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.monitoring.log_level.lower(),
    )

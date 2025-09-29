"""Command-line interface for OCAD system."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .core.config import Settings, load_config
from .core.models import Endpoint, EndpointRole
from .system.orchestrator import SystemOrchestrator

app = typer.Typer(help="ORAN CFM-Lite AI Anomaly Detection System")
console = Console()


@app.command()
def run(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    port: int = typer.Option(8080, "--port", help="API port"),
):
    """Run the OCAD system with API server."""
    console.print("[bold green]Starting ORAN CFM-Lite AI Anomaly Detection System[/bold green]")
    
    # Load configuration
    if config_file:
        settings = load_config(config_file)
    else:
        settings = Settings()
    
    # Override with CLI arguments
    settings.api_host = host
    settings.api_port = port
    
    console.print(f"API will be available at http://{host}:{port}")
    console.print("Press Ctrl+C to stop")
    
    # Start the API server
    import uvicorn
    uvicorn.run(
        "ocad.api.main:app",
        host=host,
        port=port,
        log_level=settings.monitoring.log_level.lower(),
    )


@app.command()
def add_endpoint(
    host: str = typer.Argument(..., help="Endpoint hostname or IP"),
    port: int = typer.Option(830, "--port", help="NETCONF port"),
    role: str = typer.Option("o-ru", "--role", help="Endpoint role (o-ru, o-du, transport)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
):
    """Add an endpoint for monitoring."""
    
    async def _add_endpoint():
        # Load configuration
        if config_file:
            settings = load_config(config_file)
        else:
            settings = Settings()
        
        # Create orchestrator
        orchestrator = SystemOrchestrator(settings)
        
        try:
            await orchestrator.start()
            
            # Create endpoint
            endpoint = Endpoint(
                id=f"{host}:{port}",
                host=host,
                port=port,
                role=EndpointRole(role),
            )
            
            console.print(f"Adding endpoint {endpoint.id}...")
            
            success = await orchestrator.add_endpoint(endpoint)
            
            if success:
                console.print(f"[green]✓[/green] Successfully added endpoint {endpoint.id}")
                
                # Show detected capabilities
                capabilities = orchestrator.capability_registry.get_capabilities(endpoint.id)
                if capabilities:
                    console.print("\nDetected capabilities:")
                    table = Table()
                    table.add_column("Capability")
                    table.add_column("Supported")
                    
                    table.add_row("UDP Echo", "✓" if capabilities.udp_echo else "✗")
                    table.add_row("eCPRI Delay", "✓" if capabilities.ecpri_delay else "✗")
                    table.add_row("LBM", "✓" if capabilities.lbm else "✗")
                    table.add_row("CCM Minimal", "✓" if capabilities.ccm_min else "✗")
                    table.add_row("LLDP", "✓" if capabilities.lldp else "✗")
                    
                    console.print(table)
            else:
                console.print(f"[red]✗[/red] Failed to add endpoint {endpoint.id}")
                raise typer.Exit(1)
            
        finally:
            await orchestrator.stop()
    
    asyncio.run(_add_endpoint())


@app.command()
def list_endpoints(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
):
    """List all monitored endpoints."""
    
    async def _list_endpoints():
        # Load configuration
        if config_file:
            settings = load_config(config_file)
        else:
            settings = Settings()
        
        # Create orchestrator
        orchestrator = SystemOrchestrator(settings)
        
        try:
            await orchestrator.start()
            
            endpoints = await orchestrator.list_endpoints()
            
            if not endpoints:
                console.print("No endpoints configured")
                return
            
            table = Table()
            table.add_column("ID")
            table.add_column("Host")
            table.add_column("Port")
            table.add_column("Role")
            table.add_column("Status")
            
            for endpoint in endpoints:
                status = "Active" if endpoint.active else "Inactive"
                table.add_row(
                    endpoint.id,
                    endpoint.host,
                    str(endpoint.port),
                    endpoint.role,
                    status,
                )
            
            console.print(table)
            
        finally:
            await orchestrator.stop()
    
    asyncio.run(_list_endpoints())


@app.command()
def status(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
):
    """Show system status and statistics."""
    
    async def _status():
        # Load configuration
        if config_file:
            settings = load_config(config_file)
        else:
            settings = Settings()
        
        # Create orchestrator
        orchestrator = SystemOrchestrator(settings)
        
        try:
            await orchestrator.start()
            
            # Get statistics
            stats = await orchestrator.get_statistics()
            health = await orchestrator.get_health_status()
            
            console.print("[bold]System Status[/bold]")
            console.print(f"Health: {'🟢 Healthy' if health['healthy'] else '🔴 Unhealthy'}")
            console.print(f"Uptime: {health['uptime_seconds']:.1f} seconds")
            console.print()
            
            console.print("[bold]Statistics[/bold]")
            table = Table()
            table.add_column("Metric")
            table.add_column("Value")
            
            table.add_row("Endpoints", str(stats["endpoints"]))
            table.add_row("Active Alerts", str(stats["active_alerts"]))
            table.add_row("Suppressed Endpoints", str(stats["suppressed_endpoints"]))
            table.add_row("Capability Coverage", f"{stats['capability_coverage']:.1f}%")
            table.add_row("Samples Processed", str(stats["samples_processed"]))
            table.add_row("Features Extracted", str(stats["features_extracted"]))
            table.add_row("Alerts Generated", str(stats["alerts_generated"]))
            
            if stats["processing_latency_p95"]:
                table.add_row("Processing Latency (p95)", f"{stats['processing_latency_p95']:.3f}s")
            
            console.print(table)
            
        finally:
            await orchestrator.stop()
    
    asyncio.run(_status())


@app.command()
def simulate(
    count: int = typer.Option(10, "--count", help="Number of sample endpoints to simulate"),
    duration: int = typer.Option(300, "--duration", help="Simulation duration in seconds"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
):
    """Run simulation with synthetic endpoints."""
    
    async def _simulate():
        from .utils.simulator import EndpointSimulator
        
        # Load configuration
        if config_file:
            settings = load_config(config_file)
        else:
            settings = Settings()
        
        console.print(f"[bold]Starting simulation with {count} endpoints for {duration} seconds[/bold]")
        
        # Create simulator
        simulator = EndpointSimulator(settings)
        
        try:
            await simulator.run_simulation(count, duration)
            console.print("[green]✓[/green] Simulation completed successfully")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Simulation interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]✗[/red] Simulation failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_simulate())


@app.command()
def config_example():
    """Generate example configuration file."""
    
    config_content = """# OCAD Configuration Example
# Save as config/local.yaml

environment: "development"
debug: true

api:
  host: "0.0.0.0"
  port: 8080
  workers: 1

database:
  url: "postgresql+asyncpg://ocad:ocad@localhost:5432/ocad"
  pool_size: 10

redis:
  url: "redis://localhost:6379/0"

kafka:
  bootstrap_servers: ["localhost:9092"]
  group_id: "ocad"

netconf:
  timeout: 30
  port: 830
  username: "admin"
  password: "admin"
  hostkey_verify: false

feature:
  window_size_minutes: 1
  overlap_seconds: 30
  percentiles: [95, 99]
  ewma_alpha: 0.3
  cusum_threshold: 5.0

detection:
  rule_weight: 0.35
  changepoint_weight: 0.25
  residual_weight: 0.30
  multivariate_weight: 0.10
  
  rule_timeout_ms: 5000.0
  rule_p99_threshold_ms: 100.0
  cusum_threshold: 5.0
  residual_threshold: 3.0
  
  hold_down_seconds: 120
  dedup_window_seconds: 300

alert:
  evidence_count: 3
  min_evidence_for_alert: 2
  severity_buckets:
    critical: 0.8
    warning: 0.6
    info: 0.4

monitoring:
  prometheus_port: 8000
  log_level: "INFO"
  enable_tracing: true
"""
    
    config_path = Path("config/example.yaml")
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(config_content)
    
    console.print(f"[green]✓[/green] Example configuration saved to {config_path}")
    console.print("Copy and modify this file for your environment:")
    console.print(f"  cp {config_path} config/local.yaml")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()

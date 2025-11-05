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


@app.command()
def list_plugins(
    plugin_dir: Optional[Path] = typer.Option(None, "--plugin-dir", help="Plugin directory path"),
):
    """List all available plugins (protocol adapters and detectors)."""
    from .plugins.registry import registry

    # Discover plugins
    if plugin_dir:
        plugin_path = plugin_dir
    else:
        plugin_path = Path(__file__).parent / "plugins"

    console.print(f"[bold]Discovering plugins from: {plugin_path}[/bold]\n")
    registry.discover_plugins(plugin_path)

    # List protocol adapters
    adapters = registry.list_protocol_adapters()

    if adapters:
        console.print("[bold cyan]Protocol Adapters:[/bold cyan]")
        table = Table()
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Supported Metrics")
        table.add_column("Recommended Models")

        for name, info in adapters.items():
            metrics = ", ".join(info["supported_metrics"][:3])
            if len(info["supported_metrics"]) > 3:
                metrics += f" (+{len(info['supported_metrics']) - 3} more)"

            models = ", ".join(info["recommended_models"])

            table.add_row(
                name,
                info["version"],
                metrics,
                models,
            )

        console.print(table)
        console.print()
    else:
        console.print("[yellow]No protocol adapters found[/yellow]\n")

    # List detectors
    detectors = registry.list_detectors()

    if detectors:
        console.print("[bold magenta]Detectors:[/bold magenta]")
        table = Table()
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Supported Protocols")

        for name, info in detectors.items():
            protocols = ", ".join(info["supported_protocols"])

            table.add_row(
                name,
                info["version"],
                protocols,
            )

        console.print(table)
    else:
        console.print("[yellow]No detectors found[/yellow]")


@app.command()
def plugin_info(
    name: str = typer.Argument(..., help="Plugin name (protocol adapter or detector)"),
    plugin_dir: Optional[Path] = typer.Option(None, "--plugin-dir", help="Plugin directory path"),
):
    """Show detailed information about a specific plugin."""
    from .plugins.registry import registry

    # Discover plugins
    if plugin_dir:
        plugin_path = plugin_dir
    else:
        plugin_path = Path(__file__).parent / "plugins"

    registry.discover_plugins(plugin_path)

    # Check protocol adapters
    adapter = registry.get_protocol_adapter(name)
    if adapter:
        console.print(f"[bold cyan]Protocol Adapter: {name}[/bold cyan]")
        console.print(f"Version: {adapter.version}")
        console.print(f"Description: {adapter.get_description()}")
        console.print()

        console.print("[bold]Supported Metrics:[/bold]")
        for metric in adapter.supported_metrics:
            console.print(f"  • {metric}")
        console.print()

        console.print("[bold]Recommended AI Models:[/bold]")
        for model in adapter.get_recommended_models():
            console.print(f"  • {model}")

        console.print()
        console.print("[bold]Example Configuration:[/bold]")
        console.print("""
protocol_adapters:
  {name}:
    enabled: true
    config:
      # Add protocol-specific config here
""".format(name=name))

        return

    # Check detectors
    detector = registry.get_detector(name)
    if detector:
        console.print(f"[bold magenta]Detector: {name}[/bold magenta]")
        console.print(f"Version: {detector.version}")
        console.print(f"Description: {detector.get_description()}")
        console.print()

        console.print("[bold]Supported Protocols:[/bold]")
        for protocol in detector.supported_protocols:
            console.print(f"  • {protocol}")

        console.print()
        console.print("[bold]Example Configuration:[/bold]")
        console.print("""
detectors:
  {name}:
    enabled: true
    protocols: {protocols}
    config:
      # Add detector-specific config here
""".format(name=name, protocols=detector.supported_protocols))

        return

    # Not found
    console.print(f"[red]✗[/red] Plugin '{name}' not found")
    console.print("Run 'list-plugins' to see available plugins")
    raise typer.Exit(1)


@app.command()
def enable_plugin(
    name: str = typer.Argument(..., help="Plugin name to enable"),
    config_file: Path = typer.Option("config/plugins.yaml", "--config", "-c", help="Plugin configuration file"),
):
    """Enable a plugin in the configuration file."""
    import yaml

    console.print(f"[bold]Enabling plugin: {name}[/bold]")

    # Load existing config
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {"protocol_adapters": {}, "detectors": {}}

    # Check if it's a protocol adapter or detector
    from .plugins.registry import registry
    plugin_path = Path(__file__).parent / "plugins"
    registry.discover_plugins(plugin_path)

    adapter = registry.get_protocol_adapter(name)
    detector = registry.get_detector(name)

    if adapter:
        if "protocol_adapters" not in config:
            config["protocol_adapters"] = {}

        if name not in config["protocol_adapters"]:
            config["protocol_adapters"][name] = {"enabled": True, "config": {}}
        else:
            config["protocol_adapters"][name]["enabled"] = True

        console.print(f"[green]✓[/green] Enabled protocol adapter: {name}")

    elif detector:
        if "detectors" not in config:
            config["detectors"] = {}

        if name not in config["detectors"]:
            config["detectors"][name] = {
                "enabled": True,
                "protocols": detector.supported_protocols,
                "config": {}
            }
        else:
            config["detectors"][name]["enabled"] = True

        console.print(f"[green]✓[/green] Enabled detector: {name}")

    else:
        console.print(f"[red]✗[/red] Plugin '{name}' not found")
        raise typer.Exit(1)

    # Save config
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"Configuration saved to: {config_file}")


@app.command()
def disable_plugin(
    name: str = typer.Argument(..., help="Plugin name to disable"),
    config_file: Path = typer.Option("config/plugins.yaml", "--config", "-c", help="Plugin configuration file"),
):
    """Disable a plugin in the configuration file."""
    import yaml

    console.print(f"[bold]Disabling plugin: {name}[/bold]")

    # Load existing config
    if not config_file.exists():
        console.print(f"[red]✗[/red] Configuration file not found: {config_file}")
        raise typer.Exit(1)

    with open(config_file) as f:
        config = yaml.safe_load(f) or {}

    # Find and disable
    disabled = False

    if "protocol_adapters" in config and name in config["protocol_adapters"]:
        config["protocol_adapters"][name]["enabled"] = False
        console.print(f"[green]✓[/green] Disabled protocol adapter: {name}")
        disabled = True

    if "detectors" in config and name in config["detectors"]:
        config["detectors"][name]["enabled"] = False
        console.print(f"[green]✓[/green] Disabled detector: {name}")
        disabled = True

    if not disabled:
        console.print(f"[yellow]⚠[/yellow] Plugin '{name}' not found in configuration")
        raise typer.Exit(1)

    # Save config
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"Configuration saved to: {config_file}")


@app.command()
def test_plugin(
    name: str = typer.Argument(..., help="Plugin name to test"),
    plugin_dir: Optional[Path] = typer.Option(None, "--plugin-dir", help="Plugin directory path"),
):
    """Test a specific plugin (protocol adapter or detector)."""

    async def _test_plugin():
        from .plugins.registry import registry

        # Discover plugins
        if plugin_dir:
            plugin_path = plugin_dir
        else:
            plugin_path = Path(__file__).parent / "plugins"

        console.print(f"[bold]Testing plugin: {name}[/bold]\n")
        registry.discover_plugins(plugin_path)

        # Check protocol adapters
        adapter = registry.get_protocol_adapter(name)
        if adapter:
            console.print(f"[cyan]Protocol Adapter: {adapter.name} v{adapter.version}[/cyan]")

            # Test configuration validation
            test_config = {"endpoints": [{"id": "test", "ip": "192.168.1.1"}], "interval_sec": 1}

            try:
                is_valid = adapter.validate_config(test_config)
                console.print(f"✓ Configuration validation: {is_valid}")
            except Exception as e:
                console.print(f"[red]✗[/red] Configuration validation failed: {e}")
                raise typer.Exit(1)

            # Test metric collection (5 samples)
            console.print("\nCollecting test metrics (5 samples)...")
            collected = 0

            try:
                async for metric in adapter.collect(test_config):
                    console.print(f"  • {metric['metric_name']}: {metric['value']:.2f}")
                    collected += 1
                    if collected >= 5:
                        break

                console.print(f"\n[green]✓[/green] Test passed: collected {collected} metrics")
            except Exception as e:
                console.print(f"[red]✗[/red] Test failed: {e}")
                import traceback
                traceback.print_exc()
                raise typer.Exit(1)

            return

        # Check detectors
        detector = registry.get_detector(name)
        if detector:
            console.print(f"[magenta]Detector: {detector.name} v{detector.version}[/magenta]")
            console.print(f"Supported protocols: {detector.supported_protocols}")

            # Verify methods exist
            required_methods = ["train", "detect", "save_model", "load_model"]
            for method in required_methods:
                if hasattr(detector, method):
                    console.print(f"✓ Method '{method}' exists")
                else:
                    console.print(f"[red]✗[/red] Missing method: {method}")
                    raise typer.Exit(1)

            console.print(f"\n[green]✓[/green] Test passed: all required methods present")
            return

        # Not found
        console.print(f"[red]✗[/red] Plugin '{name}' not found")
        raise typer.Exit(1)

    asyncio.run(_test_plugin())


@app.command()
def train_detector(
    name: str = typer.Argument(..., help="Detector name to train"),
    data_path: Path = typer.Option(..., "--data", help="Training data path (Parquet)"),
    output_path: Optional[Path] = typer.Option(None, "--output", help="Model output path"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for training"),
):
    """Train a detector model."""
    from .plugins.registry import registry
    import pandas as pd

    console.print(f"[bold]Training detector: {name}[/bold]\n")

    # Discover plugins
    plugin_path = Path(__file__).parent / "plugins"
    registry.discover_plugins(plugin_path)

    # Get detector
    detector = registry.get_detector(name)
    if not detector:
        console.print(f"[red]✗[/red] Detector '{name}' not found")
        raise typer.Exit(1)

    console.print(f"Detector: {detector.name} v{detector.version}")
    console.print(f"Supported protocols: {detector.supported_protocols}")

    # Load data
    if not data_path.exists():
        console.print(f"[red]✗[/red] Data file not found: {data_path}")
        raise typer.Exit(1)

    console.print(f"\nLoading training data from: {data_path}")

    try:
        data = pd.read_parquet(data_path)
        console.print(f"✓ Loaded {len(data)} samples")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load data: {e}")
        raise typer.Exit(1)

    # Train model
    console.print(f"\nTraining model (epochs={epochs}, batch_size={batch_size})...")

    try:
        detector.train(data, epochs=epochs, batch_size=batch_size)
        console.print("[green]✓[/green] Training completed")
    except Exception as e:
        console.print(f"[red]✗[/red] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

    # Save model
    if output_path:
        model_path = output_path
    else:
        model_path = Path(f"ocad/models/{name}/{name}_trained.pth")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"\nSaving model to: {model_path}")

    try:
        detector.save_model(str(model_path))
        console.print(f"[green]✓[/green] Model saved successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to save model: {e}")
        raise typer.Exit(1)


@app.command()
def detect(
    protocol: str = typer.Argument(..., help="Protocol name (cfm, bfd, bgp, ptp)"),
    endpoint: str = typer.Option(..., "--endpoint", help="Endpoint ID or IP"),
    detector_name: Optional[str] = typer.Option(None, "--detector", help="Specific detector to use"),
    duration: int = typer.Option(60, "--duration", help="Detection duration in seconds"),
):
    """Run real-time anomaly detection for a protocol endpoint."""

    async def _detect():
        from .plugins.registry import registry
        import time

        console.print(f"[bold]Starting real-time detection[/bold]")
        console.print(f"Protocol: {protocol}")
        console.print(f"Endpoint: {endpoint}")
        console.print(f"Duration: {duration} seconds\n")

        # Discover plugins
        plugin_path = Path(__file__).parent / "plugins"
        registry.discover_plugins(plugin_path)

        # Get protocol adapter
        adapter = registry.get_protocol_adapter(protocol)
        if not adapter:
            console.print(f"[red]✗[/red] Protocol adapter '{protocol}' not found")
            raise typer.Exit(1)

        # Get detectors
        if detector_name:
            detector = registry.get_detector(detector_name)
            if not detector:
                console.print(f"[red]✗[/red] Detector '{detector_name}' not found")
                raise typer.Exit(1)
            detectors = [detector]
        else:
            # Use all compatible detectors
            detectors = registry.get_detectors_for_protocol(protocol)
            if not detectors:
                console.print(f"[yellow]⚠[/yellow] No detectors found for protocol '{protocol}'")
                raise typer.Exit(1)

        console.print(f"Using detectors: {[d.name for d in detectors]}\n")

        # Configure adapter
        config = {
            "endpoints": [{"id": endpoint, "ip": endpoint}],
            "interval_sec": 1,
        }

        # Run detection
        console.print("Collecting metrics and detecting anomalies...\n")

        start_time = time.time()
        anomaly_count = 0
        sample_count = 0

        try:
            async for metric in adapter.collect(config):
                sample_count += 1

                # Run detection with each detector
                for detector in detectors:
                    try:
                        result = detector.detect(metric)

                        if result.get("is_anomaly", False):
                            anomaly_count += 1
                            console.print(
                                f"[red]🚨 ANOMALY[/red] [{detector.name}] "
                                f"{metric['metric_name']}={metric['value']:.2f} "
                                f"(score={result['score']:.2f})"
                            )
                        else:
                            console.print(
                                f"[green]✓[/green] [{detector.name}] "
                                f"{metric['metric_name']}={metric['value']:.2f} "
                                f"(score={result['score']:.2f})"
                            )

                    except Exception as e:
                        console.print(f"[yellow]⚠[/yellow] Detection failed: {e}")

                # Check duration
                if time.time() - start_time > duration:
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]Detection interrupted by user[/yellow]")

        except Exception as e:
            console.print(f"\n[red]✗[/red] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            raise typer.Exit(1)

        # Summary
        console.print("\n" + "=" * 60)
        console.print("[bold]Detection Summary[/bold]")
        console.print(f"Samples processed: {sample_count}")
        console.print(f"Anomalies detected: {anomaly_count}")

        if sample_count > 0:
            anomaly_rate = anomaly_count / sample_count * 100
            console.print(f"Anomaly rate: {anomaly_rate:.1f}%")

    asyncio.run(_detect())


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()

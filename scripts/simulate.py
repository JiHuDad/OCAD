#!/usr/bin/env python3
"""
Quick simulation script for testing OCAD system.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so we can import ocad
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.utils.simulator import EndpointSimulator


async def main():
    """Run simulation."""
    print("OCAD Simulation Test")
    print("===================")
    
    # Load configuration
    config_path = Path("config/local.yaml")
    if config_path.exists():
        from ocad.core.config import load_config
        settings = load_config(config_path)
    else:
        settings = Settings()
    
    # Create simulator
    simulator = EndpointSimulator(settings)
    
    try:
        print("Starting simulation with 5 endpoints for 120 seconds...")
        await simulator.run_simulation(endpoint_count=5, duration_seconds=120)
        print("Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

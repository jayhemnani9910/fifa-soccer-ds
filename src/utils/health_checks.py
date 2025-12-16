"""
Health checks and monitoring for FIFA Soccer DS YouTube Pipeline

This module provides health check functionality for all pipeline components
and integrates with Prometheus monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import psutil
import aiohttp
import yaml

# Configure logging
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checker for all pipeline components."""
    
    def __init__(self, config_path: str = "configs/youtube_pipeline.yaml"):
        """Initialize health checker with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.start_time = datetime.now()
        
        # Health check results cache
        self._health_cache = {}
        self._cache_ttl = 30  # 30 seconds cache
        
        # Metrics tracking
        self.metrics = {
            'health_checks_total': 0,
            'health_checks_failed': 0,
            'component_health': {},
            'last_check_time': None
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'monitoring': {
                'enabled': True,
                'metrics_endpoint': 'http://localhost:9090/metrics',
                'health_check_interval': 30
            }
        }
    
    async def check_all_components(self) -> Dict[str, Any]:
        """Run health checks on all pipeline components."""
        logger.info("Starting comprehensive health check...")
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'system_info': await self._get_system_info(),
            'metrics': self.metrics
        }
        
        # Check individual components
        components = [
            ('youtube_downloader', self._check_youtube_downloader),
            ('audio_processor', self._check_audio_processor),
            ('soccer_classifier', self._check_soccer_classifier),
            ('pipeline_orchestrator', self._check_orchestrator),
            ('api_server', self._check_api_server),
            ('monitoring', self._check_monitoring),
            ('storage', self._check_storage),
            ('dependencies', self._check_dependencies)
        ]
        
        failed_checks = 0
        
        for component_name, check_func in components:
            try:
                result = await check_func()
                health_results['components'][component_name] = result
                
                if result['status'] != 'healthy':
                    failed_checks += 1
                    
                logger.info(f"Health check - {component_name}: {result['status']}")
                
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                health_results['components'][component_name] = error_result
                failed_checks += 1
                logger.error(f"Health check failed for {component_name}: {e}")
        
        # Determine overall status
        if failed_checks == 0:
            health_results['overall_status'] = 'healthy'
        elif failed_checks < len(components) // 2:
            health_results['overall_status'] = 'degraded'
        else:
            health_results['overall_status'] = 'unhealthy'
        
        # Update metrics
        self.metrics['health_checks_total'] += 1
        if failed_checks > 0:
            self.metrics['health_checks_failed'] += 1
        self.metrics['last_check_time'] = datetime.now().isoformat()
        
        # Cache results
        self._health_cache = health_results.copy()
        self._health_cache['cached_at'] = time.time()
        
        return health_results
    
    async def _check_youtube_downloader(self) -> Dict[str, Any]:
        """Check YouTube downloader component health."""
        try:
            # Check if yt-dlp is available
            import subprocess
            result = subprocess.run(['yt-dlp', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    'status': 'healthy',
                    'message': f'yt-dlp available: {result.stdout.strip()}',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': 'yt-dlp not working properly',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to check yt-dlp: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_audio_processor(self) -> Dict[str, Any]:
        """Check audio processing component health."""
        try:
            # Check if audio processing libraries are available
            checks = []
            
            # Check whisper
            try:
                import whisper
                checks.append('whisper: available')
            except ImportError:
                checks.append('whisper: missing')
            
            # Check librosa
            try:
                import librosa
                checks.append('librosa: available')
            except ImportError:
                checks.append('librosa: missing')
            
            # Check if all required libraries are available
            missing = [check for check in checks if 'missing' in check]
            
            if not missing:
                return {
                    'status': 'healthy',
                    'message': 'All audio libraries available',
                    'details': checks,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': f'Missing audio libraries: {missing}',
                    'details': checks,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to check audio processor: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_soccer_classifier(self) -> Dict[str, Any]:
        """Check soccer classification component health."""
        try:
            # Check if classifier can be instantiated
            from ..classify.soccer_classifier import SoccerClassifier
            
            classifier = SoccerClassifier()
            
            return {
                'status': 'healthy',
                'message': 'Soccer classifier initialized successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Soccer classifier initialization failed: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_orchestrator(self) -> Dict[str, Any]:
        """Check pipeline orchestrator health."""
        try:
            # Check if orchestrator can be instantiated
            # This would be implemented when orchestrator is created
            
            return {
                'status': 'healthy',
                'message': 'Pipeline orchestrator available',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Orchestrator check failed: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_api_server(self) -> Dict[str, Any]:
        """Check API server health."""
        try:
            # Check if API server is responding
            api_config = self.config.get('api', {})
            host = api_config.get('host', '0.0.0.0')
            port = api_config.get('port', 8000)
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f'http://{host}:{port}/health') as response:
                    if response.status == 200:
                        return {
                            'status': 'healthy',
                            'message': f'API server responding on {host}:{port}',
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'message': f'API server returned status {response.status}',
                            'timestamp': datetime.now().isoformat()
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'API server health check failed: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_monitoring(self) -> Dict[str, Any]:
        """Check monitoring system health."""
        try:
            monitoring_config = self.config.get('monitoring', {})
            metrics_endpoint = monitoring_config.get('metrics_endpoint', 'http://localhost:9090/metrics')
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(metrics_endpoint) as response:
                    if response.status == 200:
                        return {
                            'status': 'healthy',
                            'message': f'Monitoring endpoint responding: {metrics_endpoint}',
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'message': f'Monitoring endpoint returned status {response.status}',
                            'timestamp': datetime.now().isoformat()
                        }
                        
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Monitoring endpoint not available: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_storage(self) -> Dict[str, Any]:
        """Check storage and disk space."""
        try:
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Check if we have enough space (at least 10GB free)
            if free_gb >= 10:
                status = 'healthy'
                message = f'Disk space OK: {free_gb:.1f}GB free of {total_gb:.1f}GB'
            elif free_gb >= 5:
                status = 'warning'
                message = f'Low disk space: {free_gb:.1f}GB free'
            else:
                status = 'unhealthy'
                message = f'Critical disk space: {free_gb:.1f}GB free'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'free_gb': round(free_gb, 1),
                    'total_gb': round(total_gb, 1),
                    'usage_percent': round(usage_percent, 1)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Storage check failed: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies and requirements."""
        try:
            dependencies = [
                ('python', 'python --version'),
                ('ffmpeg', 'ffmpeg -version'),
                ('git', 'git --version')
            ]
            
            results = []
            missing_deps = []
            
            for dep_name, cmd in dependencies:
                try:
                    result = subprocess.run(cmd.split(), 
                                          capture_output=True, text=True, 
                                          timeout=10)
                    if result.returncode == 0:
                        results.append(f'{dep_name}: available')
                    else:
                        results.append(f'{dep_name}: error')
                        missing_deps.append(dep_name)
                except Exception:
                    results.append(f'{dep_name}: missing')
                    missing_deps.append(dep_name)
            
            if not missing_deps:
                status = 'healthy'
                message = 'All system dependencies available'
            else:
                status = 'warning'
                message = f'Missing dependencies: {missing_deps}'
            
            return {
                'status': status,
                'message': message,
                'details': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Dependency check failed: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'process_count': len(psutil.pids()),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_cached_health(self) -> Optional[Dict[str, Any]]:
        """Get cached health check results if still valid."""
        if not self._health_cache:
            return None
        
        cached_at = self._health_cache.get('cached_at', 0)
        if time.time() - cached_at < self._cache_ttl:
            return self._health_cache
        return None
    
    async def get_quick_status(self) -> Dict[str, Any]:
        """Get a quick status check (uses cache if available)."""
        cached = self.get_cached_health()
        if cached:
            return cached
        
        # If no cache, do a quick check
        return await self.check_all_components()


# Health check decorator for automatic monitoring
def health_check(func):
    """Decorator to automatically add health check monitoring to functions."""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            
            # Log successful execution
            duration = time.time() - start_time
            logger.info(f"Function {func.__name__} completed successfully in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            # Log failed execution
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper


# Background health monitoring
class BackgroundHealthMonitor:
    """Background service for continuous health monitoring."""
    
    def __init__(self, health_checker: HealthChecker, interval: int = 60):
        self.health_checker = health_checker
        self.interval = interval
        self.running = False
        self.task = None
    
    async def start(self):
        """Start background health monitoring."""
        self.running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("Background health monitoring started")
    
    async def stop(self):
        """Stop background health monitoring."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Background health monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                await self.health_checker.check_all_components()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background health check failed: {e}")
                await asyncio.sleep(self.interval)


if __name__ == "__main__":
    # Test health checker
    async def main():
        checker = HealthChecker()
        health = await checker.check_all_components()
        print(f"Overall health: {health['overall_status']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status['status']}")
    
    asyncio.run(main())
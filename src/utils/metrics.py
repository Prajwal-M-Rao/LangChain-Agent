"""
Metrics collection and monitoring for the Enhanced Wikipedia Agent.
Tracks performance, usage, and system metrics.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict, deque
import psutil
from dataclasses import dataclass, asdict


@dataclass
class MetricEntry:
    """Represents a single metric entry."""
    timestamp: float
    metric_type: str
    name: str
    value: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_entries: Maximum number of entries to keep in memory
        """
        self.max_entries = max_entries
        self.metrics: deque = deque(maxlen=max_entries)
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def record_metric(self, metric_type: str, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric entry.
        
        Args:
            metric_type: Type of metric (counter, gauge, histogram, timer)
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        with self.lock:
            entry = MetricEntry(
                timestamp=time.time(),
                metric_type=metric_type,
                name=name,
                value=value,
                metadata=metadata or {}
            )
            self.metrics.append(entry)
            
            # Update aggregated metrics
            if metric_type == "counter":
                self.counters[name] += value
            elif metric_type == "gauge":
                self.gauges[name] = value
            elif metric_type == "histogram":
                self.histograms[name].append(value)
            elif metric_type == "timer":
                self.histograms[f"{name}_duration"].append(value)
                
    def increment_counter(self, name: str, value: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Increment a counter metric."""
        self.record_metric("counter", name, value, metadata)
        
    def set_gauge(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Set a gauge metric."""
        self.record_metric("gauge", name, value, metadata)
        
    def record_histogram(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a histogram value."""
        self.record_metric("histogram", name, value, metadata)
        
    def start_timer(self, name: str) -> str:
        """Start a timer and return a timer ID."""
        timer_id = f"{name}_{time.time()}"
        self.timers[timer_id] = time.time()
        return timer_id
        
    def end_timer(self, timer_id: str, metadata: Optional[Dict[str, Any]] = None):
        """End a timer and record the duration."""
        if timer_id in self.timers:
            duration = time.time() - self.timers[timer_id]
            name = timer_id.split('_')[0]
            self.record_metric("timer", name, duration, metadata)
            del self.timers[timer_id]
            return duration
        return None
        
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
        
    def get_gauge(self, name: str) -> float:
        """Get gauge value."""
        return self.gauges.get(name, 0.0)
        
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {}
            
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "median": sorted_values[count // 2],
            "p95": sorted_values[int(count * 0.95)] if count > 0 else 0,
            "p99": sorted_values[int(count * 0.99)] if count > 0 else 0,
        }
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            }
        except Exception as e:
            return {"error": str(e)}
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {name: self.get_histogram_stats(name) for name in self.histograms},
                "system": self.get_system_metrics(),
                "total_entries": len(self.metrics)
            }
            return summary
            
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        summary = self.get_metrics_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()


class PerformanceMonitor:
    """Context manager for performance monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            operation_name: Name of the operation being monitored
            metadata: Additional metadata
        """
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
        self.timer_id = None
        
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.timer_id = self.metrics_collector.start_timer(self.operation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and record metrics."""
        if self.timer_id:
            duration = self.metrics_collector.end_timer(self.timer_id, self.metadata)
            
            # Record additional metrics
            if exc_type is not None:
                self.metrics_collector.increment_counter(f"{self.operation_name}_errors")
            else:
                self.metrics_collector.increment_counter(f"{self.operation_name}_success")


class AgentMetrics:
    """High-level metrics interface for the Wikipedia Agent."""
    
    def __init__(self):
        """Initialize agent metrics."""
        self.collector = MetricsCollector()
        
    def record_query(self, query_type: str, confidence: float, sources_used: List[str]):
        """Record a query event."""
        self.collector.increment_counter("queries_total")
        self.collector.increment_counter(f"queries_{query_type}")
        self.collector.record_histogram("query_confidence", confidence)
        
        for source in sources_used:
            self.collector.increment_counter(f"source_{source}_used")
            
    def record_response_time(self, operation: str, duration: float):
        """Record response time."""
        self.collector.record_histogram(f"{operation}_response_time", duration)
        
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.collector.increment_counter(f"cache_{cache_type}_hits")
        
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.collector.increment_counter(f"cache_{cache_type}_misses")
        
    def record_api_call(self, api_name: str, success: bool, response_time: float):
        """Record API call metrics."""
        self.collector.increment_counter(f"api_{api_name}_calls")
        if success:
            self.collector.increment_counter(f"api_{api_name}_success")
        else:
            self.collector.increment_counter(f"api_{api_name}_errors")
        self.collector.record_histogram(f"api_{api_name}_response_time", response_time)
        
    def record_memory_usage(self, component: str, usage_mb: float):
        """Record memory usage."""
        self.collector.set_gauge(f"memory_{component}_mb", usage_mb)
        
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.collector.increment_counter(f"errors_{error_type}")
        self.collector.increment_counter(f"errors_{component}")
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        summary = self.collector.get_metrics_summary()
        
        # Calculate additional derived metrics
        total_queries = summary["counters"].get("queries_total", 0)
        total_errors = sum(v for k, v in summary["counters"].items() if k.startswith("errors_"))
        
        if total_queries > 0:
            error_rate = (total_errors / total_queries) * 100
        else:
            error_rate = 0
            
        summary["derived_metrics"] = {
            "error_rate_percent": error_rate,
            "total_queries": total_queries,
            "total_errors": total_errors,
        }
        
        return summary
        
    def monitor_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a performance monitor for an operation."""
        return PerformanceMonitor(self.collector, operation_name, metadata)
        
    def export_report(self, filepath: str):
        """Export performance report to file."""
        report = self.get_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
    def reset(self):
        """Reset all metrics."""
        self.collector.reset_metrics()


# Global metrics instance
metrics = AgentMetrics()

# Convenience functions
def record_query(query_type: str, confidence: float, sources_used: List[str]):
    """Record a query event."""
    metrics.record_query(query_type, confidence, sources_used)

def record_response_time(operation: str, duration: float):
    """Record response time."""
    metrics.record_response_time(operation, duration)

def record_cache_hit(cache_type: str):
    """Record cache hit."""
    metrics.record_cache_hit(cache_type)

def record_cache_miss(cache_type: str):
    """Record cache miss."""
    metrics.record_cache_miss(cache_type)

def record_api_call(api_name: str, success: bool, response_time: float):
    """Record API call."""
    metrics.record_api_call(api_name, success, response_time)

def record_error(error_type: str, component: str):
    """Record error."""
    metrics.record_error(error_type, component)

def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Monitor an operation."""
    return metrics.monitor_operation(operation_name, metadata)

def get_performance_report() -> Dict[str, Any]:
    """Get performance report."""
    return metrics.get_performance_report()

def export_report(filepath: str):
    """Export performance report."""
    metrics.export_report(filepath)
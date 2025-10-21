"""
P2P Performance Dashboard for ExLlamaV3

This module provides a web-based dashboard for monitoring and visualizing P2P GPU communication
performance metrics in real-time.
"""

import os
import sys
import json
import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import numpy as np

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import flask
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available. Dashboard will run in CLI mode.")

try:
    import plotly
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Charts will be disabled.")

from exllamav3.util.p2p_monitor import get_global_monitor, P2PMonitor
from exllamav3.util.p2p_profiler import get_global_profiler, P2PProfiler
from exllamav3.util.p2p_debug import get_global_debugger, P2PDebugger


class P2PPerformanceDashboard:
    """
    Web-based dashboard for P2P performance monitoring.
    
    This class provides a Flask-based web interface for real-time monitoring
    and visualization of P2P GPU communication performance.
    """
    
    def __init__(
        self,
        monitor: Optional[P2PMonitor] = None,
        profiler: Optional[P2PProfiler] = None,
        debugger: Optional[P2PDebugger] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        debug: bool = False
    ):
        """
        Initialize P2P performance dashboard.
        
        Args:
            monitor: P2P monitor instance
            profiler: P2P profiler instance
            debugger: P2P debugger instance
            host: Host address for web server
            port: Port for web server
            debug: Enable Flask debug mode
        """
        self.monitor = monitor or get_global_monitor()
        self.profiler = profiler or get_global_profiler()
        self.debugger = debugger or get_global_debugger()
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app if available
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self._setup_routes()
        else:
            self.app = None
        
        # Dashboard state
        self.running = False
        self.server_thread = None
        
        # Performance data cache
        self.performance_cache = {
            "last_update": 0,
            "device_metrics": {},
            "operation_stats": {},
            "topology_info": {},
            "recent_events": [],
            "bottlenecks": []
        }
        self.cache_lock = threading.Lock()
        
        # Real-time data
        self.real_time_data = {
            "bandwidth_history": defaultdict(lambda: deque(maxlen=100)),
            "latency_history": defaultdict(lambda: deque(maxlen=100)),
            "memory_history": defaultdict(lambda: deque(maxlen=100)),
            "operation_history": defaultdict(lambda: deque(maxlen=100))
        }
        self.data_lock = threading.Lock()
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""
        if not self.app:
            return
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('p2p_dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current performance metrics."""
            self._update_performance_cache()
            return jsonify(self.performance_cache)
        
        @self.app.route('/api/device/<int:device_id>')
        def get_device_metrics(device_id):
            """Get metrics for a specific device."""
            if self.monitor:
                device_metrics = self.monitor.get_device_metrics(device_id)
                if device_metrics:
                    return jsonify({
                        "device_id": device_id,
                        "metrics": device_metrics.__dict__
                    })
            return jsonify({"error": "Device not found"})
        
        @self.app.route('/api/topology')
        def get_topology():
            """Get topology information."""
            if self.monitor and self.monitor.topology:
                topology_summary = self.monitor.topology.get_topology_summary()
                p2p_matrix = self.monitor.topology.get_p2p_matrix()
                
                return jsonify({
                    "summary": topology_summary,
                    "p2p_matrix": p2p_matrix.tolist() if p2p_matrix.size > 0 else [],
                    "tree_stats": self.monitor.topology.get_tree_stats(
                        self.monitor.topology.communication_tree
                    ) if self.monitor.topology.communication_tree else {}
                })
            return jsonify({"error": "Topology not available"})
        
        @self.app.route('/api/operations')
        def get_operations():
            """Get recent operations."""
            limit = request.args.get('limit', 50, type=int)
            operation_type = request.args.get('type', None)
            
            if self.monitor:
                operations = self.monitor.get_operation_history(
                    operation_type=operation_type,
                    limit=limit
                )
                
                return jsonify({
                    "operations": [op.__dict__ for op in operations]
                })
            return jsonify({"operations": []})
        
        @self.app.route('/api/bottlenecks')
        def get_bottlenecks():
            """Get performance bottlenecks."""
            if self.monitor:
                bottlenecks = self.monitor.identify_bottlenecks()
                suggestions = self.monitor.get_optimization_suggestions()
                
                return jsonify({
                    "bottlenecks": bottlenecks,
                    "suggestions": suggestions
                })
            return jsonify({"bottlenecks": [], "suggestions": []})
        
        @self.app.route('/api/debug')
        def get_debug_info():
            """Get debugging information."""
            if self.debugger:
                debug_stats = self.debugger.get_statistics()
                recent_errors = self.debugger.get_errors(limit=10)
                
                return jsonify({
                    "statistics": debug_stats,
                    "recent_errors": [err.__dict__ for err in recent_errors]
                })
            return jsonify({"statistics": {}, "recent_errors": []})
        
        @self.app.route('/api/charts/bandwidth')
        def get_bandwidth_chart():
            """Get bandwidth chart data."""
            if not PLOTLY_AVAILABLE:
                return jsonify({"error": "Plotly not available"})
            
            chart_data = self._create_bandwidth_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/charts/latency')
        def get_latency_chart():
            """Get latency chart data."""
            if not PLOTLY_AVAILABLE:
                return jsonify({"error": "Plotly not available"})
            
            chart_data = self._create_latency_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/charts/memory')
        def get_memory_chart():
            """Get memory usage chart data."""
            if not PLOTLY_AVAILABLE:
                return jsonify({"error": "Plotly not available"})
            
            chart_data = self._create_memory_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/charts/topology')
        def get_topology_chart():
            """Get topology visualization chart data."""
            if not PLOTLY_AVAILABLE:
                return jsonify({"error": "Plotly not available"})
            
            chart_data = self._create_topology_chart()
            return jsonify(chart_data)
    
    def _update_performance_cache(self):
        """Update performance metrics cache."""
        with self.cache_lock:
            current_time = time.time()
            
            # Update cache every second
            if current_time - self.performance_cache["last_update"] < 1.0:
                return
            
            # Get device metrics
            if self.monitor:
                device_metrics = self.monitor.get_all_device_metrics()
                self.performance_cache["device_metrics"] = {
                    device_id: metrics.__dict__ 
                    for device_id, metrics in device_metrics.items()
                }
                
                # Get operation statistics
                summary = self.monitor.get_performance_summary()
                self.performance_cache["operation_stats"] = summary.get("operation_stats", {})
                
                # Get topology information
                if self.monitor.topology_metrics:
                    self.performance_cache["topology_info"] = self.monitor.topology_metrics.__dict__
                
                # Get recent events
                recent_events = self.monitor.get_operation_history(limit=20)
                self.performance_cache["recent_events"] = [
                    event.__dict__ for event in recent_events
                ]
                
                # Get bottlenecks
                bottlenecks = self.monitor.identify_bottlenecks()
                self.performance_cache["bottlenecks"] = bottlenecks
            
            self.performance_cache["last_update"] = current_time
    
    def _create_bandwidth_chart(self) -> Dict[str, Any]:
        """Create bandwidth chart data."""
        if not PLOTLY_AVAILABLE or not self.monitor:
            return {"error": "Chart generation not available"}
        
        # Get bandwidth data from operations
        operations = self.monitor.get_operation_history(limit=100)
        
        # Group by device pair
        bandwidth_data = defaultdict(list)
        for op in operations:
            if op.success and op.bandwidth_gbps > 0:
                if op.peer_device is not None:
                    pair = f"{op.device_id} <-> {op.peer_device}"
                else:
                    pair = f"Device {op.device_id}"
                
                bandwidth_data[pair].append({
                    "time": op.end_time,
                    "bandwidth": op.bandwidth_gbps
                })
        
        # Create chart traces
        traces = []
        for pair, data in bandwidth_data.items():
            if data:
                times = [d["time"] for d in data]
                bandwidths = [d["bandwidth"] for d in data]
                
                trace = go.Scatter(
                    x=times,
                    y=bandwidths,
                    mode='lines+markers',
                    name=pair,
                    line=dict(width=2)
                )
                traces.append(trace)
        
        layout = go.Layout(
            title='P2P Bandwidth Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Bandwidth (GB/s)'),
            hovermode='closest'
        )
        
        figure = go.Figure(data=traces, layout=layout)
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(figure))
    
    def _create_latency_chart(self) -> Dict[str, Any]:
        """Create latency chart data."""
        if not PLOTLY_AVAILABLE or not self.monitor:
            return {"error": "Chart generation not available"}
        
        # Get latency data from operations
        operations = self.monitor.get_operation_history(limit=100)
        
        # Group by operation type
        latency_data = defaultdict(list)
        for op in operations:
            if op.success:
                latency_data[op.operation_type].append({
                    "time": op.end_time,
                    "latency": op.duration_ms
                })
        
        # Create chart traces
        traces = []
        for op_type, data in latency_data.items():
            if data:
                times = [d["time"] for d in data]
                latencies = [d["latency"] for d in data]
                
                trace = go.Scatter(
                    x=times,
                    y=latencies,
                    mode='lines+markers',
                    name=op_type,
                    line=dict(width=2)
                )
                traces.append(trace)
        
        layout = go.Layout(
            title='P2P Operation Latency Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Latency (ms)'),
            hovermode='closest'
        )
        
        figure = go.Figure(data=traces, layout=layout)
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(figure))
    
    def _create_memory_chart(self) -> Dict[str, Any]:
        """Create memory usage chart data."""
        if not PLOTLY_AVAILABLE or not self.monitor:
            return {"error": "Chart generation not available"}
        
        # Get memory data from device metrics
        device_metrics = self.monitor.get_all_device_metrics()
        
        traces = []
        for device_id, metrics in device_metrics.items():
            if metrics.gpu_memory_total > 0:
                used_mb = metrics.gpu_memory_used / (1024**2)
                total_mb = metrics.gpu_memory_total / (1024**2)
                
                trace = go.Bar(
                    x=[f"Device {device_id}"],
                    y=[used_mb],
                    name=f"Device {device_id} Used",
                    text=[f"{used_mb:.1f} MB / {total_mb:.1f} MB"]
                )
                traces.append(trace)
        
        layout = go.Layout(
            title='GPU Memory Usage by Device',
            xaxis=dict(title='Device'),
            yaxis=dict(title='Memory Usage (MB)'),
            showlegend=False
        )
        
        figure = go.Figure(data=traces, layout=layout)
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(figure))
    
    def _create_topology_chart(self) -> Dict[str, Any]:
        """Create topology visualization chart data."""
        if not PLOTLY_AVAILABLE or not self.monitor or not self.monitor.topology:
            return {"error": "Topology visualization not available"}
        
        topology = self.monitor.topology
        p2p_matrix = topology.get_p2p_matrix()
        
        # Create node positions
        num_devices = topology.num_devices
        angle_step = 2 * np.pi / num_devices
        
        node_trace = go.Scatter(
            x=[np.cos(i * angle_step) for i in range(num_devices)],
            y=[np.sin(i * angle_step) for i in range(num_devices)],
            mode='markers+text',
            text=[f"GPU {topology.active_devices[i]}" for i in range(num_devices)],
            textposition="middle center",
            marker=dict(size=20, color='lightblue'),
            name='Devices'
        )
        
        # Create edges for P2P connections
        edge_x = []
        edge_y = []
        
        for i in range(num_devices):
            for j in range(num_devices):
                if i != j and p2p_matrix[i][j]:
                    x1, y1 = np.cos(i * angle_step), np.sin(i * angle_step)
                    x2, y2 = np.cos(j * angle_step), np.sin(j * angle_step)
                    edge_x.extend([x1, x2, None])
                    edge_y.extend([y1, y2, None])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2, color='gray'),
            name='P2P Connections'
        )
        
        layout = go.Layout(
            title='P2P Topology Visualization',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest'
        )
        
        figure = go.Figure(data=[edge_trace, node_trace], layout=layout)
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(figure))
    
    def start(self):
        """Start the dashboard web server."""
        if not FLASK_AVAILABLE:
            print("Flask not available. Cannot start web dashboard.")
            return
        
        if self.running:
            print("Dashboard already running.")
            return
        
        self.running = True
        
        def run_server():
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,
                threaded=True
            )
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        print(f"P2P Performance Dashboard started at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the dashboard web server."""
        if not self.running:
            return
        
        self.running = False
        
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        
        print("P2P Performance Dashboard stopped.")
    
    def run_cli_dashboard(self):
        """Run a command-line dashboard."""
        print("P2P Performance Dashboard (CLI Mode)")
        print("=" * 50)
        
        while True:
            try:
                self._update_performance_cache()
                
                # Display device metrics
                print("\nDevice Metrics:")
                print("-" * 30)
                for device_id, metrics in self.performance_cache["device_metrics"].items():
                    print(f"Device {device_id}:")
                    print(f"  Operations: {metrics.get('total_operations', 0)}")
                    print(f"  Success Rate: {metrics.get('successful_operations', 0) / max(metrics.get('total_operations', 1), 1) * 100:.1f}%")
                    print(f"  Avg Bandwidth: {metrics.get('average_bandwidth_gbps', 0):.2f} GB/s")
                    print(f"  Memory Usage: {metrics.get('gpu_memory_used', 0) / (1024**3):.2f} GB")
                
                # Display operation statistics
                print("\nOperation Statistics:")
                print("-" * 30)
                for op_type, stats in self.performance_cache["operation_stats"].items():
                    print(f"{op_type}: {stats.get('count', 0)} operations, "
                          f"{stats.get('success_rate_percent', 0):.1f}% success")
                
                # Display bottlenecks
                bottlenecks = self.performance_cache.get("bottlenecks", {})
                if any(bottlenecks.values()):
                    print("\nBottlenecks:")
                    print("-" * 30)
                    for category, items in bottlenecks.items():
                        if items:
                            print(f"{category}: {len(items)} issues")
                
                print("\nPress Ctrl+C to exit...")
                time.sleep(5.0)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5.0)


def create_dashboard_template():
    """Create HTML template for the dashboard."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'p2p_dashboard.html')
    
    if os.path.exists(template_path):
        return  # Template already exists
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P2P Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .refresh-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>P2P Performance Dashboard</h1>
            <p>Real-time monitoring of P2P GPU communication</p>
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
        
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated here -->
        </div>
        
        <div class="chart-container">
            <h2>Bandwidth Over Time</h2>
            <div id="bandwidth-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Latency Over Time</h2>
            <div id="latency-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Memory Usage</h2>
            <div id="memory-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Topology Visualization</h2>
            <div id="topology-chart"></div>
        </div>
    </div>

    <script>
        function refreshData() {
            // Refresh metrics
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => updateMetrics(data));
            
            // Refresh charts
            fetch('/api/charts/bandwidth')
                .then(response => response.json())
                .then(data => Plotly.newPlot('bandwidth-chart', data.data, data.layout));
            
            fetch('/api/charts/latency')
                .then(response => response.json())
                .then(data => Plotly.newPlot('latency-chart', data.data, data.layout));
            
            fetch('/api/charts/memory')
                .then(response => response.json())
                .then(data => Plotly.newPlot('memory-chart', data.data, data.layout));
            
            fetch('/api/charts/topology')
                .then(response => response.json())
                .then(data => Plotly.newPlot('topology-chart', data.data, data.layout));
        }
        
        function updateMetrics(data) {
            const metricsGrid = document.getElementById('metrics-grid');
            metricsGrid.innerHTML = '';
            
            // Device metrics
            for (const [deviceId, metrics] of Object.entries(data.device_metrics)) {
                const card = createMetricCard(
                    `Device ${deviceId}`,
                    [
                        `Operations: ${metrics.total_operations}`,
                        `Success Rate: ${metrics.successful_operations / Math.max(metrics.total_operations, 1) * 100}%`,
                        `Avg Bandwidth: ${metrics.average_bandwidth_gbps.toFixed(2)} GB/s`,
                        `Memory: ${(metrics.gpu_memory_used / 1024**3).toFixed(2)} GB`
                    ]
                );
                metricsGrid.appendChild(card);
            }
            
            // Operation stats
            for (const [opType, stats] of Object.entries(data.operation_stats)) {
                const card = createMetricCard(
                    opType,
                    [
                        `Count: ${stats.count}`,
                        `Success Rate: ${stats.success_rate_percent.toFixed(1)}%`
                    ]
                );
                metricsGrid.appendChild(card);
            }
        }
        
        function createMetricCard(title, items) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            
            const titleElement = document.createElement('h3');
            titleElement.textContent = title;
            card.appendChild(titleElement);
            
            const list = document.createElement('ul');
            items.forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                list.appendChild(li);
            });
            card.appendChild(list);
            
            return card;
        }
        
        // Initial load
        refreshData();
        
        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
"""
    
    with open(template_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard template created at {template_path}")


def main():
    """Main function for running the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="P2P Performance Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create dashboard template
    create_dashboard_template()
    
    # Initialize dashboard
    dashboard = P2PPerformanceDashboard(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    if args.cli or not FLASK_AVAILABLE:
        # Run CLI dashboard
        dashboard.run_cli_dashboard()
    else:
        # Run web dashboard
        dashboard.start()
        
        try:
            print("Dashboard running. Press Ctrl+C to stop.")
            while dashboard.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        
        dashboard.stop()


if __name__ == "__main__":
    main()
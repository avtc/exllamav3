"""
P2P Topology Visualizer for ExLlamaV3

This module provides visualization and analysis tools for P2P GPU communication topologies,
including interactive graphs, performance heatmaps, and topology optimization suggestions.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Advanced topology analysis will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Static visualizations will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive visualizations will be disabled.")

from exllamav3.model.model_tp_p2p import P2PTopology


class P2PTopologyVisualizer:
    """
    Comprehensive P2P topology visualization and analysis tool.
    
    This class provides various visualization methods for P2P topologies,
    including static graphs, interactive plots, and performance analysis.
    """
    
    def __init__(
        self,
        topology: P2PTopology,
        output_dir: str = "./topology_visualizations"
    ):
        """
        Initialize topology visualizer.
        
        Args:
            topology: P2P topology object
            output_dir: Directory to save visualizations
        """
        self.topology = topology
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance data (if available)
        self.performance_data = {}
        
        # Visualization settings
        self.node_colors = {
            'default': 'lightblue',
            'root': 'lightgreen',
            'leaf': 'lightcoral',
            'high_traffic': 'orange',
            'bottleneck': 'red'
        }
        
        self.edge_colors = {
            'default': 'gray',
            'high_bandwidth': 'green',
            'low_bandwidth': 'red',
            'high_latency': 'orange'
        }
    
    def set_performance_data(self, performance_data: Dict[str, Any]):
        """Set performance data for topology visualization."""
        self.performance_data = performance_data
    
    def create_static_topology_graph(
        self,
        layout: str = "circular",
        show_labels: bool = True,
        show_performance: bool = False,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create static topology graph using Matplotlib.
        
        Args:
            layout: Graph layout algorithm ("circular", "spring", "kamada_kawai")
            show_labels: Show device labels
            show_performance: Show performance information
            save_path: Path to save the visualization
            
        Returns:
            Path to saved visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for static visualizations")
        
        # Create NetworkX graph
        G = self._create_networkx_graph()
        
        # Set layout
        if layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.circular_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            if show_performance and 'bandwidth' in data:
                # Color edges by bandwidth
                bandwidth = data['bandwidth']
                if bandwidth > 20:  # High bandwidth
                    edge_colors.append(self.edge_colors['high_bandwidth'])
                elif bandwidth < 5:  # Low bandwidth
                    edge_colors.append(self.edge_colors['low_bandwidth'])
                else:
                    edge_colors.append(self.edge_colors['default'])
                
                # Width by bandwidth
                edge_widths.append(min(5, max(1, bandwidth / 10)))
            else:
                edge_colors.append(self.edge_colors['default'])
                edge_widths.append(2)
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7
        )
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node, data in G.nodes(data=True):
            # Determine node color based on role
            if data.get('is_root', False):
                node_colors.append(self.node_colors['root'])
            elif data.get('is_leaf', False):
                node_colors.append(self.node_colors['leaf'])
            elif show_performance and 'traffic' in data:
                if data['traffic'] > 100:  # High traffic
                    node_colors.append(self.node_colors['high_traffic'])
                else:
                    node_colors.append(self.node_colors['default'])
            else:
                node_colors.append(self.node_colors['default'])
            
            # Node size based on degree or performance
            if show_performance and 'performance_score' in data:
                node_sizes.append(300 + data['performance_score'] * 10)
            else:
                node_sizes.append(300 + G.degree(node) * 50)
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw labels
        if show_labels:
            labels = {node: f"GPU {node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)
        
        # Add title and legend
        topology_type = self.topology.topology_type or "unknown"
        ax.set_title(f"P2P Topology Visualization ({topology_type})", fontsize=16)
        
        # Create legend
        legend_elements = []
        if show_performance:
            legend_elements.append(
                patches.Patch(color=self.edge_colors['high_bandwidth'], label='High Bandwidth')
            )
            legend_elements.append(
                patches.Patch(color=self.edge_colors['low_bandwidth'], label='Low Bandwidth')
            )
            legend_elements.append(
                patches.Patch(color=self.node_colors['high_traffic'], label='High Traffic')
            )
        
        legend_elements.append(
            patches.Patch(color=self.node_colors['root'], label='Root Node')
        )
        legend_elements.append(
            patches.Patch(color=self.node_colors['leaf'], label='Leaf Node')
        )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Add topology information
        info_text = self._get_topology_info_text()
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save visualization
        if not save_path:
            timestamp = int(time.time())
            save_path = os.path.join(self.output_dir, f"topology_graph_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_topology_graph(
        self,
        show_performance: bool = False,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create interactive topology graph using Plotly.
        
        Args:
            show_performance: Show performance information
            save_path: Path to save the visualization
            
        Returns:
            Path to saved visualization
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations")
        
        # Create NetworkX graph
        G = self._create_networkx_graph()
        
        # Calculate positions
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge information for hover
            info = f"GPU {u} ↔ GPU {v}"
            if show_performance and 'bandwidth' in data:
                info += f"<br>Bandwidth: {data['bandwidth']:.2f} GB/s"
            if show_performance and 'latency' in data:
                info += f"<br>Latency: {data['latency']:.2f} ms"
            edge_info.append(info)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text
            node_text.append(f"GPU {node}")
            
            # Node information for hover
            info = f"GPU {node}<br>Degree: {G.degree(node)}"
            if data.get('is_root', False):
                info += "<br>Role: Root"
            elif data.get('is_leaf', False):
                info += "<br>Role: Leaf"
            
            if show_performance:
                if 'traffic' in data:
                    info += f"<br>Traffic: {data['traffic']:.2f}"
                if 'performance_score' in data:
                    info += f"<br>Performance: {data['performance_score']:.2f}"
            
            node_info.append(info)
            
            # Node color
            if data.get('is_root', False):
                node_colors.append('lightgreen')
            elif data.get('is_leaf', False):
                node_colors.append('lightcoral')
            elif show_performance and 'traffic' in data and data['traffic'] > 100:
                node_colors.append('orange')
            else:
                node_colors.append('lightblue')
            
            # Node size
            if show_performance and 'performance_score' in data:
                node_sizes.append(20 + data['performance_score'])
            else:
                node_sizes.append(20 + G.degree(node) * 2)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f"P2P Topology ({self.topology.topology_type or 'unknown'})",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[
                               dict(
                                   text=self._get_topology_info_text(),
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=0.995,
                                   xanchor='left', yanchor='top',
                                   font=dict(size=10),
                                   bgcolor="rgba(255,255,255,0.8)"
                               )
                           ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # Save visualization
        if not save_path:
            timestamp = int(time.time())
            save_path = os.path.join(self.output_dir, f"topology_interactive_{timestamp}.html")
        
        fig.write_html(save_path)
        
        return save_path
    
    def create_performance_heatmap(
        self,
        metric: str = "bandwidth",
        save_path: Optional[str] = None
    ) -> str:
        """
        Create performance heatmap for device pairs.
        
        Args:
            metric: Performance metric to visualize ("bandwidth", "latency", "success_rate")
            save_path: Path to save the visualization
            
        Returns:
            Path to saved visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for heatmaps")
        
        # Get performance matrix
        performance_matrix = self._get_performance_matrix(metric)
        
        if performance_matrix is None:
            raise ValueError(f"No performance data available for metric: {metric}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap
        if metric == "bandwidth":
            cmap = 'Greens'
            label = 'Bandwidth (GB/s)'
        elif metric == "latency":
            cmap = 'Reds'
            label = 'Latency (ms)'
        elif metric == "success_rate":
            cmap = 'Blues'
            label = 'Success Rate (%)'
        else:
            cmap = 'viridis'
            label = metric
        
        im = ax.imshow(performance_matrix, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        device_ids = self.topology.active_devices
        ax.set_xticks(np.arange(len(device_ids)))
        ax.set_yticks(np.arange(len(device_ids)))
        ax.set_xticklabels([f"GPU {d}" for d in device_ids])
        ax.set_yticklabels([f"GPU {d}" for d in device_ids])
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, rotation=270, labelpad=20)
        
        # Add title
        ax.set_title(f"P2P Performance Heatmap ({metric})", fontsize=16)
        
        # Add values to cells
        for i in range(len(device_ids)):
            for j in range(len(device_ids)):
                if i == j:
                    continue
                
                value = performance_matrix[i, j]
                if metric == "success_rate":
                    text = f"{value:.1f}"
                else:
                    text = f"{value:.2f}"
                
                ax.text(j, i, text, ha="center", va="center", color="black" if value < np.max(performance_matrix) / 2 else "white")
        
        plt.tight_layout()
        
        # Save visualization
        if not save_path:
            timestamp = int(time.time())
            save_path = os.path.join(self.output_dir, f"performance_heatmap_{metric}_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_topology_analysis_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive topology analysis report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        # Analyze topology
        analysis = self._analyze_topology()
        
        # Create report
        report = {
            "timestamp": time.time(),
            "topology_info": self.topology.get_topology_summary(),
            "analysis": analysis,
            "recommendations": self._generate_topology_recommendations(analysis)
        }
        
        # Save report
        if not save_path:
            timestamp = int(time.time())
            save_path = os.path.join(self.output_dir, f"topology_analysis_{timestamp}.json")
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return save_path
    
    def _create_networkx_graph(self) -> nx.Graph:
        """Create NetworkX graph from P2P topology."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for graph creation")
        
        G = nx.Graph()
        
        # Add nodes
        for device_id in self.topology.active_devices:
            node_data = {"device_id": device_id}
            
            # Add role information
            if self.topology.communication_tree:
                tree = self.topology.communication_tree
                if tree.get("root") == device_id:
                    node_data["is_root"] = True
                if device_id not in tree.get("children", {}):
                    node_data["is_leaf"] = True
            
            # Add performance data
            if device_id in self.performance_data:
                node_data.update(self.performance_data[device_id])
            
            G.add_node(device_id, **node_data)
        
        # Add edges
        p2p_matrix = self.topology.get_p2p_matrix()
        for i, device_i in enumerate(self.topology.active_devices):
            for j, device_j in enumerate(self.topology.active_devices):
                if i != j and p2p_matrix[i][j]:
                    edge_data = {"p2p_available": True}
                    
                    # Add performance data
                    pair_key = tuple(sorted([device_i, device_j]))
                    if pair_key in self.performance_data:
                        edge_data.update(self.performance_data[pair_key])
                    
                    G.add_edge(device_i, device_j, **edge_data)
        
        return G
    
    def _get_topology_info_text(self) -> str:
        """Get topology information text for visualizations."""
        summary = self.topology.get_topology_summary()
        
        info = f"Devices: {summary.get('num_devices', 0)}\n"
        info += f"Topology: {summary.get('topology_type', 'unknown')}\n"
        info += f"Connectivity: {summary.get('connectivity_ratio', 0):.2f}\n"
        info += f"Fully Connected: {summary.get('is_fully_connected', False)}"
        
        if self.topology.communication_tree:
            tree_stats = self.topology.get_tree_stats(self.topology.communication_tree)
            info += f"\nTree Depth: {tree_stats.get('tree_depth', 0)}"
            info += f"\nAvg Branching: {tree_stats.get('avg_branching_factor', 0):.2f}"
        
        return info
    
    def _get_performance_matrix(self, metric: str) -> Optional[np.ndarray]:
        """Get performance matrix for a specific metric."""
        if not self.performance_data:
            return None
        
        num_devices = len(self.topology.active_devices)
        matrix = np.zeros((num_devices, num_devices))
        
        for i, device_i in enumerate(self.topology.active_devices):
            for j, device_j in enumerate(self.topology.active_devices):
                if i == j:
                    matrix[i, j] = 0  # Diagonal
                    continue
                
                pair_key = tuple(sorted([device_i, device_j]))
                if pair_key in self.performance_data:
                    pair_data = self.performance_data[pair_key]
                    if metric in pair_data:
                        matrix[i, j] = pair_data[metric]
        
        return matrix
    
    def _analyze_topology(self) -> Dict[str, Any]:
        """Analyze topology characteristics."""
        analysis = {
            "basic_metrics": {},
            "connectivity_analysis": {},
            "performance_analysis": {},
            "topology_optimization": {}
        }
        
        # Basic metrics
        summary = self.topology.get_topology_summary()
        analysis["basic_metrics"] = {
            "num_devices": summary.get("num_devices", 0),
            "topology_type": summary.get("topology_type", "unknown"),
            "connectivity_ratio": summary.get("connectivity_ratio", 0),
            "is_fully_connected": summary.get("is_fully_connected", False)
        }
        
        # Connectivity analysis
        if NETWORKX_AVAILABLE:
            G = self._create_networkx_graph()
            
            analysis["connectivity_analysis"] = {
                "average_degree": np.mean([d for n, d in G.degree()]),
                "max_degree": max([d for n, d in G.degree()]),
                "min_degree": min([d for n, d in G.degree()]),
                "density": nx.density(G),
                "is_connected": nx.is_connected(G)
            }
            
            # Centrality measures
            analysis["connectivity_analysis"]["betweenness_centrality"] = nx.betweenness_centrality(G)
            analysis["connectivity_analysis"]["closeness_centrality"] = nx.closeness_centrality(G)
            analysis["connectivity_analysis"]["degree_centrality"] = nx.degree_centrality(G)
        
        # Performance analysis
        if self.performance_data:
            # Calculate average performance metrics
            bandwidths = []
            latencies = []
            
            for key, data in self.performance_data.items():
                if isinstance(key, tuple):  # Device pair
                    if "bandwidth" in data:
                        bandwidths.append(data["bandwidth"])
                    if "latency" in data:
                        latencies.append(data["latency"])
            
            if bandwidths:
                analysis["performance_analysis"]["avg_bandwidth"] = np.mean(bandwidths)
                analysis["performance_analysis"]["max_bandwidth"] = np.max(bandwidths)
                analysis["performance_analysis"]["min_bandwidth"] = np.min(bandwidths)
            
            if latencies:
                analysis["performance_analysis"]["avg_latency"] = np.mean(latencies)
                analysis["performance_analysis"]["max_latency"] = np.max(latencies)
                analysis["performance_analysis"]["min_latency"] = np.min(latencies)
        
        # Topology optimization
        if self.topology.communication_tree:
            tree_stats = self.topology.get_tree_stats(self.topology.communication_tree)
            analysis["topology_optimization"] = {
                "tree_depth": tree_stats.get("tree_depth", 0),
                "avg_branching_factor": tree_stats.get("avg_branching_factor", 0),
                "max_branching_factor": tree_stats.get("max_branching_factor", 0),
                "num_leaves": tree_stats.get("num_leaves", 0)
            }
        
        return analysis
    
    def _generate_topology_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate topology optimization recommendations."""
        recommendations = []
        
        # Connectivity recommendations
        connectivity = analysis["connectivity_analysis"]
        if connectivity.get("density", 0) < 0.5:
            recommendations.append("Consider enabling more P2P connections to improve topology density")
        
        if connectivity.get("is_connected", True) == False:
            recommendations.append("Topology is not fully connected - this may impact performance")
        
        # Performance recommendations
        perf_analysis = analysis.get("performance_analysis", {})
        if "avg_bandwidth" in perf_analysis and perf_analysis["avg_bandwidth"] < 10:
            recommendations.append("Low average bandwidth detected - consider optimizing communication patterns")
        
        if "avg_latency" in perf_analysis and perf_analysis["avg_latency"] > 10:
            recommendations.append("High average latency detected - consider reducing communication overhead")
        
        # Topology optimization recommendations
        topo_opt = analysis.get("topology_optimization", {})
        if topo_opt.get("tree_depth", 0) > 5:
            recommendations.append("Deep tree topology - consider reducing tree depth for better latency")
        
        if topo_opt.get("avg_branching_factor", 0) < 2:
            recommendations.append("Low branching factor - consider using higher branching factor for better scalability")
        
        return recommendations


def main():
    """Main function for topology visualization."""
    parser = argparse.ArgumentParser(description="P2P Topology Visualizer")
    parser.add_argument("--devices", nargs="+", type=int, required=True,
                        help="List of device IDs")
    parser.add_argument("--output-dir", default="./topology_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--static", action="store_true",
                        help="Create static visualizations")
    parser.add_argument("--interactive", action="store_true",
                        help="Create interactive visualizations")
    parser.add_argument("--heatmap", action="store_true",
                        help="Create performance heatmaps")
    parser.add_argument("--analysis", action="store_true",
                        help="Create topology analysis report")
    parser.add_argument("--all", action="store_true",
                        help="Create all visualizations")
    
    args = parser.parse_args()
    
    # Create topology
    topology = P2PTopology(args.devices)
    
    # Create visualizer
    visualizer = P2PTopologyVisualizer(topology, args.output_dir)
    
    # Generate visualizations
    if args.all or args.static:
        try:
            path = visualizer.create_static_topology_graph()
            print(f"Static topology graph saved to: {path}")
        except Exception as e:
            print(f"Failed to create static graph: {e}")
    
    if args.all or args.interactive:
        try:
            path = visualizer.create_interactive_topology_graph()
            print(f"Interactive topology graph saved to: {path}")
        except Exception as e:
            print(f"Failed to create interactive graph: {e}")
    
    if args.all or args.heatmap:
        try:
            path = visualizer.create_performance_heatmap("bandwidth")
            print(f"Bandwidth heatmap saved to: {path}")
        except Exception as e:
            print(f"Failed to create heatmap: {e}")
    
    if args.all or args.analysis:
        try:
            path = visualizer.create_topology_analysis_report()
            print(f"Topology analysis report saved to: {path}")
        except Exception as e:
            print(f"Failed to create analysis report: {e}")
    
    print("Topology visualization complete!")


if __name__ == "__main__":
    main()
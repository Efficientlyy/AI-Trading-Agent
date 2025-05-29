import * as d3 from 'd3';
import React, { useEffect, useRef, useState } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { Agent } from '../../context/SystemControlContext';
import AgentCard from './AgentCard';

// Define a type to store root instances for cleanup
type NodeRoots = Map<string, Root>;

interface D3AgentFlowProps {
  agents: Agent[];
  onStartAgent: (agentId: string) => void;
  onStopAgent: (agentId: string) => void;
}

const D3AgentFlow: React.FC<D3AgentFlowProps> = ({ agents, onStartAgent, onStopAgent }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  // Use a Map to store root instances for proper cleanup
  const [nodeRoots] = useState<NodeRoots>(new Map());

  // Setup dimensions
  const margin = { top: 40, right: 40, bottom: 40, left: 40 };
  const width = 1200 - margin.left - margin.right;
  const height = 600 - margin.top - margin.bottom;

  // Create force simulation
  useEffect(() => {
    if (!containerRef.current || agents.length === 0) return;

    // Clean up previous visualization first
    // Clear existing roots to prevent race conditions
    nodeRoots.forEach(root => {
      try {
        root.unmount();
      } catch (e) {
        console.log('Root already unmounted');
      }
    });
    nodeRoots.clear();

    // Remove existing SVG
    if (svgRef.current) {
      d3.select(svgRef.current).remove();
      svgRef.current = null;
    }

    // Clear any existing content
    if (containerRef.current) {
      // Keep the ref but remove children
      const container = d3.select(containerRef.current);
      container.selectAll('svg').remove();
      container.selectAll('div').remove();
    }

    // Create nodes for each agent
    const nodes = agents.map((agent, i) => ({
      id: agent.agent_id,
      agent,
      x: 150 + (i % 3) * 320,
      y: 100 + Math.floor(i / 3) * 280,
      width: 300,
      height: 240
    }));

    // Create links based on agent inputs_from and outputs_to
    const links: { source: string; target: string; value: number }[] = [];
    nodes.forEach(sourceNode => {
      if (sourceNode.agent.outputs_to) {
        sourceNode.agent.outputs_to.forEach(targetId => {
          // Ensure the target node exists
          if (nodes.some(n => n.id === targetId)) {
            links.push({
              source: sourceNode.id,
              target: targetId,
              value: 1 // Default value, can be customized
            });
          } else {
            console.warn(`D3AgentFlow: Target agent ID "${targetId}" not found for source "${sourceNode.id}".`);
          }
        });
      }
      // Optionally, handle inputs_from if you want to draw arrows in the other direction
      // or ensure bidirectional links are represented. For now, outputs_to is sufficient for directed graph.
    });

    // Create the SVG container
    const svg = d3.select(containerRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .style('position', 'absolute')
      .style('z-index', 1)
      .style('pointer-events', 'none')
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    svgRef.current = svg.node()?.parentNode as SVGSVGElement;

    // Create a overlay container for React components
    const reactContainer = d3.select(containerRef.current)
      .append('div')
      .style('position', 'relative')
      .style('z-index', 2);

    // Add agent cards as React components with React 18 createRoot API
    nodes.forEach((node) => {
      // Create container for the agent card
      const nodeContainer = reactContainer
        .append('div')
        .style('position', 'absolute')
        .style('left', `${node.x}px`)
        .style('top', `${node.y}px`)
        .node();

      // Skip if null (shouldn't happen, but TypeScript needs this check)
      if (!nodeContainer) return;

      // Use modern React 18 createRoot API with proper type checking
      const root = createRoot(nodeContainer);
      root.render(
        <AgentCard
          id={node.id as string} // D3 node id might be number | string
          selected={false} // D3 nodes don't have React Flow's selection state here
          data={{
            agent: node.agent,
            onStart: onStartAgent,
            onStop: onStopAgent
          }}
        />
      );

      // Store the root in our Map for proper cleanup
      nodeRoots.set(node.id, root);
    });

    // Create the links (arrows)
    svg.append('g')
      .attr('class', 'links')
      .selectAll('path')
      .data(links)
      .enter()
      .append('path')
      .attr('d', (d) => {
        const sourceNode = nodes.find(n => n.id === d.source);
        const targetNode = nodes.find(n => n.id === d.target);

        if (!sourceNode || !targetNode) return '';

        // Calculate line with offset to account for card dimensions
        const sourceX = sourceNode.x + sourceNode.width / 2;
        const sourceY = sourceNode.y + sourceNode.height / 2;
        const targetX = targetNode.x + targetNode.width / 2;
        const targetY = targetNode.y + targetNode.height / 2;

        return `M${sourceX},${sourceY} L${targetX},${targetY}`;
      })
      .attr('stroke', '#6ff')
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead)')
      .attr('stroke-dasharray', '5,5')
      .attr('opacity', 0.7);

    // Define arrow markers
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5 L10,0 L0,5')
      .attr('fill', '#6ff');

    // Add grid background pattern
    svg.append('defs')
      .append('pattern')
      .attr('id', 'grid')
      .attr('width', 20)
      .attr('height', 20)
      .attr('patternUnits', 'userSpaceOnUse')
      .append('path')
      .attr('d', 'M 20 0 L 0 0 0 20')
      .attr('fill', 'none')
      .attr('stroke', '#232b3b')
      .attr('stroke-width', 0.5);

    // Add background with pattern
    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'url(#grid)')
      .attr('transform', `translate(0, 0)`)
      .attr('opacity', 0.4)
      .lower();

    // Clean up function
    return () => {
      // Use the stored root map for proper unmounting (React 18 way)
      nodeRoots.forEach(root => {
        root.unmount();
      });

      // Clear the map after unmounting
      nodeRoots.clear();

      // Remove the SVG element
      if (svgRef.current) {
        d3.select(svgRef.current).remove();
      }
    };
  }, [agents, onStartAgent, onStopAgent, width, height]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '70vh',
        position: 'relative',
        overflow: 'hidden',
        background: '#151a24',
        borderRadius: 12,
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
      }}
    >
      {/* D3 visualization will be rendered here */}
    </div>
  );
};

export default D3AgentFlow;
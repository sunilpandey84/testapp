import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import networkx as nx
from typing import Dict, List, Any
import logging

# Import your existing lineage system
# Make sure to adjust the import path based on your project structure
try:
    from lineageAgentFinal_HTL import LineageOrchestrator, LineageRequest
except ImportError:
    # If direct import fails, you might need to adjust the path
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from __init__ import LineageOrchestrator, LineageRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="🔍 Data Lineage Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }

    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = LineageOrchestrator()

    if 'current_result' not in st.session_state:
        st.session_state.current_result = None

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    if 'awaiting_feedback' not in st.session_state:
        st.session_state.awaiting_feedback = False

    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = None


def create_network_graph(nodes: List[Dict], edges: List[Dict]) -> go.Figure:
    """Create an interactive network graph using Plotly"""
    if not nodes or not edges:
        fig = go.Figure()
        fig.add_annotation(text="No data to visualize",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    # Create NetworkX graph for layout
    G = nx.DiGraph()

    # Add nodes
    for node in nodes:
        G.add_node(node['id'], **node)

    # Add edges
    for edge in edges:
        if G.has_node(edge['source']) and G.has_node(edge['target']):
            G.add_edge(edge['source'], edge['target'], **edge)

    # Calculate layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50)
    except:
        # Fallback layout if spring_layout fails
        pos = {node['id']: (i, 0) for i, node in enumerate(nodes)}

    # Prepare edge traces
    edge_x, edge_y = [], []
    edge_info = []

    for edge in edges:
        if edge['source'] in pos and edge['target'] in pos:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"Transformation: {edge.get('transformation', 'N/A')}")

    # Prepare node traces
    node_x = [pos[node['id']][0] for node in nodes if node['id'] in pos]
    node_y = [pos[node['id']][1] for node in nodes if node['id'] in pos]
    node_text = [f"{node['name']}<br>Table: {node['table']}" for node in nodes if node['id'] in pos]
    node_colors = ['lightblue' if node['type'] == 'source' else 'lightcoral'
                   for node in nodes if node['id'] in pos]

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=20, color=node_colors, line=dict(width=2, color='black')),
        text=[node['name'] for node in nodes if node['id'] in pos],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title="Data Lineage Network Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(text="Blue: Source nodes, Red: Target nodes",
                 showarrow=False, xref="paper", yref="paper",
                 x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                 font=dict(color="gray", size=12))
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    return fig


def display_lineage_results(result: Dict[str, Any]):
    """Display comprehensive lineage analysis results"""

    # Executive Summary
    if result.get('executive_summary'):
        st.markdown("### 📊 Executive Summary")
        st.markdown(f'<div class="info-box">{result["executive_summary"]}</div>',
                    unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lineage Type", result.get('lineage_type', 'N/A').replace('_', ' ').title())
    with col2:
        st.metric("Data Elements", len(result.get('nodes', [])))
    with col3:
        st.metric("Relationships", len(result.get('edges', [])))
    with col4:
        complexity = result.get('complexity_score', 0)
        st.metric("Complexity Score", f"{complexity}/10")

    # Network Visualization
    if result.get('nodes') and result.get('edges'):
        st.markdown("### 🕸️ Lineage Network Visualization")
        fig = create_network_graph(result['nodes'], result['edges'])
        st.plotly_chart(fig, use_container_width=True)

    # Data Flow Details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Data Elements")
        if result.get('nodes'):
            df_nodes = pd.DataFrame(result['nodes'])
            df_nodes = df_nodes[['name', 'table', 'type']].rename(columns={
                'name': 'Element Name',
                'table': 'Table',
                'type': 'Type'
            })
            st.dataframe(df_nodes, use_container_width=True)
        else:
            st.info("No data elements found")

    with col2:
        st.markdown("### 🔄 Transformations")
        if result.get('edges'):
            df_edges = pd.DataFrame(result['edges'])
            if 'transformation' in df_edges.columns:
                df_transformations = df_edges[['source', 'target', 'transformation']].rename(columns={
                    'source': 'Source',
                    'target': 'Target',
                    'transformation': 'Transformation Rule'
                })
                st.dataframe(df_transformations, use_container_width=True)
            else:
                st.dataframe(df_edges, use_container_width=True)
        else:
            st.info("No transformations found")

    # Detailed Mappings
    if result.get('query_results', {}).get('mappings'):
        st.markdown("### 🗺️ Detailed Element Mappings")
        mappings = result['query_results']['mappings']
        df_mappings = pd.DataFrame(mappings)

        if not df_mappings.empty:
            # Select and rename columns for better display
            display_columns = ['source_name', 'source_table', 'target_name', 'target_table', 'rules', 'query_code']
            available_columns = [col for col in display_columns if col in df_mappings.columns]

            if available_columns:
                df_display = df_mappings[available_columns].rename(columns={
                    'source_name': 'Source Element',
                    'source_table': 'Source Table',
                    'target_name': 'Target Element',
                    'target_table': 'Target Table',
                    'rules': 'Transformation Rules',
                    'query_code': 'Query Code'
                })
                st.dataframe(df_display, use_container_width=True)
            else:
                st.dataframe(df_mappings, use_container_width=True)

    # Recommendations
    if result.get('recommendations'):
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.markdown(f"{i}. {rec}")

    # Raw Results (Expandable)
    with st.expander("🔍 View Raw Results"):
        st.json(result)


async def process_query_async(query: str) -> Dict[str, Any]:
    """Process query asynchronously"""
    request = LineageRequest(query=query)
    return await st.session_state.orchestrator.execute_lineage_request(request)


async def resume_with_feedback_async(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Resume workflow with feedback asynchronously"""
    return await st.session_state.orchestrator.resume_with_feedback(feedback)


def handle_human_feedback(feedback_data: Dict[str, Any]):
    """Handle human feedback interface"""
    st.markdown("### 🤝 Human Input Required")

    message = feedback_data.get('message', 'Please provide input.')
    st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)

    query_results = feedback_data.get('query_results', {})

    # Handle different types of feedback requests
    if 'available_elements' in query_results:
        elements = query_results['available_elements']
        st.markdown("**Available Data Elements:**")

        # Create radio buttons for selection
        element_options = [f"{elem['name']}" for elem in elements]
        selected_element = st.radio("Choose an element:", element_options, key="element_selection")

        if st.button("Submit Selection", type="primary"):
            selected_index = element_options.index(selected_element)
            feedback = {"selected_index": selected_index}

            with st.spinner("Processing your selection..."):
                try:
                    result = asyncio.run(resume_with_feedback_async(feedback))
                    st.session_state.current_result = result
                    st.session_state.awaiting_feedback = False
                    st.session_state.feedback_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing feedback: {str(e)}")

    elif 'available_contracts' in query_results:
        contracts = query_results['available_contracts']
        st.markdown("**Available Data Contracts:**")

        # Create radio buttons for selection
        contract_options = [f"{contract['name']}" for contract in contracts]
        selected_contract = st.radio("Choose a contract:", contract_options, key="contract_selection")

        if st.button("Submit Selection", type="primary"):
            selected_index = contract_options.index(selected_contract)
            feedback = {"selected_index": selected_index}

            with st.spinner("Processing your selection..."):
                try:
                    result = asyncio.run(resume_with_feedback_async(feedback))
                    st.session_state.current_result = result
                    st.session_state.awaiting_feedback = False
                    st.session_state.feedback_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing feedback: {str(e)}")

    elif 'ambiguous_elements' in query_results:
        elements = query_results['ambiguous_elements']
        st.markdown("**Multiple elements found. Please choose one:**")

        # Create selection interface for ambiguous elements
        element_options = [f"{elem['element_name']} (Code: {elem['element_code']})" for elem in elements]
        selected_element = st.radio("Choose the correct element:", element_options, key="ambiguous_selection")

        if st.button("Submit Selection", type="primary"):
            selected_index = element_options.index(selected_element)
            feedback = {"selected_index": selected_index}

            with st.spinner("Processing your selection..."):
                try:
                    result = asyncio.run(resume_with_feedback_async(feedback))
                    st.session_state.current_result = result
                    st.session_state.awaiting_feedback = False
                    st.session_state.feedback_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing feedback: {str(e)}")


def main():
    """Main Streamlit application"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🔍 Intelligent Data Lineage Explorer</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Query Tools")

        # Example queries
        st.markdown("#### 💡 Example Queries")
        example_queries = [
            "trace customer_id lineage",
            "show Customer Data Pipeline contract",
            "upstream dependencies for order_total",
            "bidirectional lineage for email_address"
        ]

        for query in example_queries:
            if st.button(f"📝 {query}", key=f"example_{query}"):
                st.session_state.query_input = query

        # Query history
        if st.session_state.query_history:
            st.markdown("#### 📚 Query History")
            for i, (timestamp, query) in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"🕒 {query[:30]}...", key=f"history_{i}"):
                    st.session_state.query_input = query

        # Settings
        st.markdown("#### ⚙️ Settings")
        max_depth = st.slider("Max Tracing Depth", 1, 10, 5)
        show_complexity = st.checkbox("Show Complexity Analysis", True)
        auto_refresh = st.checkbox("Auto-refresh Results", False)

    # Main content area
    if st.session_state.awaiting_feedback and st.session_state.feedback_data:
        # Handle human feedback
        handle_human_feedback(st.session_state.feedback_data)
    else:
        # Regular query interface
        st.markdown("### 🔍 Query Interface")

        # Query input
        query_input = st.text_area(
            "Enter your data lineage query:",
            value=st.session_state.get('query_input', ''),
            height=100,
            placeholder="e.g., 'trace customer_id lineage' or 'show Customer Data Pipeline contract'"
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            analyze_button = st.button("🚀 Analyze Lineage", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            if st.session_state.current_result:
                export_button = st.button("💾 Export", use_container_width=True)

        # Handle buttons
        if clear_button:
            st.session_state.current_result = None
            st.session_state.query_input = ''
            st.rerun()

        if analyze_button and query_input:
            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.query_history.append((timestamp, query_input))

            with st.spinner("🤖 Analyzing data lineage with AI agents..."):
                try:
                    result = asyncio.run(process_query_async(query_input))

                    if result.get('human_input_required'):
                        # Handle human input requirement
                        st.session_state.awaiting_feedback = True
                        st.session_state.feedback_data = result
                        st.rerun()
                    else:
                        # Display results
                        st.session_state.current_result = result
                        st.session_state.awaiting_feedback = False
                        st.session_state.feedback_data = None

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    logger.error(f"Query processing error: {e}")

        # Display results
        if st.session_state.current_result and not st.session_state.awaiting_feedback:
            st.markdown("---")
            result = st.session_state.current_result

            if result.get('success') == False or result.get('error'):
                # Error handling
                error_msg = result.get('error', 'Unknown error occurred')
                st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {error_msg}</div>',
                            unsafe_allow_html=True)

                if result.get('suggestions'):
                    st.markdown("**Suggestions:**")
                    for suggestion in result['suggestions']:
                        st.markdown(f"• {suggestion}")
            else:
                # Success - display results
                st.markdown('<div class="success-box">✅ <strong>Analysis Complete!</strong></div>',
                            unsafe_allow_html=True)
                display_lineage_results(result)

        elif not st.session_state.awaiting_feedback and not st.session_state.current_result:
            # Welcome message
            st.markdown("### 👋 Welcome to Data Lineage Explorer")
            st.markdown("""
            This intelligent system uses AI agents to analyze your data lineage queries. Here's what you can do:

            **🎯 Query Types:**
            - **Element-based**: Trace specific data fields (e.g., "trace customer_id")
            - **Contract-based**: Analyze data pipelines (e.g., "show Customer Pipeline")

            **🔍 Query Examples:**
            - `trace email_address lineage`
            - `upstream dependencies for order_total`
            - `show Customer Data Pipeline contract`
            - `bidirectional lineage for user_id`

            **🤖 AI Features:**
            - Natural language query processing
            - Intelligent disambiguation
            - Context-aware recommendations
            - Interactive clarifications

            Enter your query above to get started! 🚀
            """)


if __name__ == "__main__":
    main()
from a2a_project.claude_llm.a2a_executor import LineageA2AOrchestrator


async def main():
    """Enhanced example usage with proper error handling"""

    lineage_orchestrator = LineageA2AOrchestrator()

    try:
        # Start the system
        await lineage_orchestrator.start_system()

        # Start a lineage analysis
        workflow_id = await lineage_orchestrator.start_lineage_analysis(
            "show me lineage for customer_id element lineage",
            user_id="analyst_1"
        )

        print(f"Started workflow: {workflow_id}")

        # Wait for completion with timeout
        result = await lineage_orchestrator.wait_for_completion(workflow_id, timeout=120.0)

        if result['status'] == 'completed':
            print("Analysis completed!")
            results = result.get('data', {}).get('results', {})
            print(f"Executive Summary: {results.get('executive_summary', 'No summary available')}")

            # Print key metrics
            metrics = results.get('metrics', {})
            if metrics:
                print(f"\nKey Metrics:")
                print(f"- Elements Traced: {metrics.get('elements_traced', 'N/A')}")
                print(f"- Data Sources: {metrics.get('data_sources', 'N/A')}")
                print(f"- Complexity Score: {results.get('complexity_score', 'N/A')}/10")

        elif result['status'] == 'error':
            print(f"Analysis failed: {result.get('error', 'Unknown error')}")

        elif result['status'] == 'timeout':
            print(f"Analysis timed out: {result.get('error', 'Timeout occurred')}")

        # Handle human input scenarios
        elif result.get('data', {}).get('human_input_required'):
            print("Human input required:")
            print(result['data']['message'])

            # Example: automatically select first option for demo
            if 'options' in result['data']:
                feedback = {'selected_index': 0}
                success = await lineage_orchestrator.handle_human_feedback(workflow_id, feedback)
                if success:
                    print("Feedback provided, waiting for completion...")
                    # Wait again for final result
                    final_result = await lineage_orchestrator.wait_for_completion(workflow_id, timeout=60.0)
                    print(f"Final result: {final_result['status']}")

    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always shutdown gracefully
        await lineage_orchestrator.shutdown()
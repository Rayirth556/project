"""
demo_risk_workflow.py
==================
Quick demo of the simplified risk workflow.
"""

from simplified_risk_workflow import SimplifiedRiskWorkflow

def main():
    """Run demo without interactive input"""
    print("BANK RISK ASSESSMENT DEMO")
    print("=" * 50)
    
    workflow = SimplifiedRiskWorkflow()
    
    # Run demo with sample clients from your transaction data
    print("Running demo with 3 sample clients from transactions_data.csv...")
    workflow.run_batch_workflow(client_ids=[7475327, 561, 1129], max_clients=3)

if __name__ == "__main__":
    main()

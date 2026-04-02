import logging

logger = logging.getLogger(__name__)

def select_best_model(run_results: list) -> dict:
    """
    Selects the run with the lowest RMSE.
    
    run_results is a list of dicts: [{'run_name': str, 'metrics': dict, 'params': dict}]
    """
    if not run_results:
        return {}
        
    best_run = min(run_results, key=lambda x: x['metrics']['rmse'])
    
    logger.info(f"🏆 Best Model: {best_run['run_name']} with RMSE: {best_run['metrics']['rmse']:.2f}")
    return best_run

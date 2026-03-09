"""
A/B Testing API Endpoints for FinColl

Provides REST API for managing A/B test experiments:
- Start/stop experiments
- Get experiment status
- View variant assignments
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..ab_testing_manager import ab_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/experiments", tags=["ab_testing"])


class StartExperimentRequest(BaseModel):
    """Request to start A/B test experiment"""
    experiment_id: str
    control_model: str
    treatment_model: str
    traffic_split: float = 0.1  # Default: 10% to treatment
    status: str = "running"


class ExperimentStatusResponse(BaseModel):
    """Experiment status response"""
    active: bool
    experiment: Optional[Dict] = None


class VariantAssignmentRequest(BaseModel):
    """Request to check variant assignment"""
    symbol: str


class VariantAssignmentResponse(BaseModel):
    """Variant assignment response"""
    symbol: str
    variant: str
    model_version: str
    metadata: Dict


@router.post("/start")
async def start_experiment(request: StartExperimentRequest):
    """
    Start A/B test experiment

    Args:
        experiment_id: Unique experiment identifier
        control_model: Control model version (e.g., 'epoch45_2026-02-01')
        treatment_model: Treatment model version (e.g., 'epoch50_2026-02-08')
        traffic_split: Fraction of traffic to treatment (0.0-1.0)

    Returns:
        Success confirmation with experiment details
    """
    try:
        # Validate traffic split
        if not 0.0 <= request.traffic_split <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="traffic_split must be between 0.0 and 1.0"
            )

        # Load experiment into A/B manager
        ab_manager.load_experiment({
            'experiment_id': request.experiment_id,
            'control_model': request.control_model,
            'treatment_model': request.treatment_model,
            'traffic_split': request.traffic_split,
            'status': request.status
        })

        logger.info(
            f"📊 A/B Test Started: {request.experiment_id} "
            f"({request.traffic_split * 100:.0f}% traffic to treatment)"
        )

        return {
            'success': True,
            'experiment_id': request.experiment_id,
            'message': f'A/B test experiment started: {request.traffic_split * 100:.0f}% traffic to treatment',
            'experiment': ab_manager.get_experiment_status()
        }

    except Exception as e:
        logger.error(f"Failed to start experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_experiment():
    """
    Stop active A/B test experiment

    Returns:
        Success confirmation
    """
    try:
        if not ab_manager.active_experiment:
            return {
                'success': True,
                'message': 'No active experiment to stop'
            }

        experiment_id = ab_manager.active_experiment.experiment_id
        ab_manager.stop_experiment()

        logger.info(f"📊 A/B Test Stopped: {experiment_id}")

        return {
            'success': True,
            'experiment_id': experiment_id,
            'message': 'A/B test experiment stopped'
        }

    except Exception as e:
        logger.error(f"Failed to stop experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_experiment_status() -> ExperimentStatusResponse:
    """
    Get active experiment status

    Returns:
        Experiment details if active, or None
    """
    experiment = ab_manager.get_experiment_status()

    return ExperimentStatusResponse(
        active=experiment is not None,
        experiment=experiment
    )


@router.post("/assign-variant")
async def assign_variant(request: VariantAssignmentRequest) -> VariantAssignmentResponse:
    """
    Get variant assignment for a symbol

    Useful for debugging and understanding variant distribution.

    Args:
        symbol: Stock symbol to check

    Returns:
        Variant assignment details
    """
    variant, model_version = ab_manager.assign_variant(request.symbol)
    metadata = ab_manager.get_variant_metadata(request.symbol)

    return VariantAssignmentResponse(
        symbol=request.symbol,
        variant=variant,
        model_version=model_version,
        metadata=metadata
    )

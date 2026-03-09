"""
Prediction Engine for FinColl
Wraps FinVec model for production inference with OHLC data
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import sys

# Add finvec to path
finvec_path = Path(__file__).parent.parent.parent.parent / "finvec"
sys.path.insert(0, str(finvec_path))

from models.financial_llm import FinancialLLM
from configs.model_config import FinancialLLMConfig
from data.tokenizers.financial_tokenizer import FinancialTokenizer


class PredictionEngine:
    """
    Production inference engine for FinVec model
    Handles OHLC data tokenization and prediction extraction
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize prediction engine

        Args:
            checkpoint_path: Path to FinVec checkpoint
            device: 'cpu', 'cuda', or 'cuda:0'
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        print(f"[PredictionEngine] Loading model from {checkpoint_path}")
        print(f"[PredictionEngine] Using device: {device}")

        # Load checkpoint
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Create tokenizer
        self.tokenizer = FinancialTokenizer()
        vocab_size = self.tokenizer.get_vocab_size()

        # Detect if checkpoint was trained with continuous horizons
        # Check config first, then check model_state_dict for range_heads
        checkpoint_config = self.checkpoint.get('config', {})
        config_has_continuous = checkpoint_config.get('use_continuous_horizons', False)
        
        # Check if model has range_heads (continuous architecture)
        model_state_dict = self.checkpoint.get('model_state_dict', {})
        has_range_heads = any('range_heads' in key for key in model_state_dict.keys())
        
        # Use continuous if either config says so OR model has range_heads
        self.use_continuous_horizons = config_has_continuous or has_range_heads
        
        if has_range_heads and not config_has_continuous:
            print(f"[PredictionEngine] Detected continuous horizons from model architecture (range_heads found)")

        # Create model config
        self.config = FinancialLLMConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=12,
            d_ff=2048,
            max_seq_length=2048,
            dropout=0.1,
            prediction_horizons=[1, 5, 20],
            prediction_horizon_names=["1d", "5d", "20d"],
            # Enable prediction heads (required for continuous or discrete)
            price_prediction=True,
            volatility_prediction=True,
            regime_classification=True,
            risk_scoring=True,
            use_continuous_horizons=self.use_continuous_horizons,
            horizon_ranges=[(1, 20), (21, 50), (51, 100)] if self.use_continuous_horizons else None
        )

        if self.use_continuous_horizons:
            print(f"[PredictionEngine] Using CONTINUOUS horizon predictions (1-100 days)")

        # Create and load model
        self.model = FinancialLLM(self.config)
        self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[PredictionEngine] Model loaded: {param_count:.1f}M parameters")
        print(f"[PredictionEngine] Epoch: {self.checkpoint['epoch']}, Step: {self.checkpoint['step']}")

    def predict(self, ohlc_data: np.ndarray, symbol: str = "UNKNOWN") -> Dict:
        """
        Run inference on OHLC data

        Args:
            ohlc_data: numpy array [n_bars, 5] with columns [Open, High, Low, Close, Volume]
            symbol: Symbol name for logging

        Returns:
            Dictionary with predictions:
            {
                'symbol': str,
                'direction': 'LONG' or 'SHORT',
                'confidence': float (0-100),
                'predictions': {
                    '1d': float,
                    '5d': float,
                    '20d': float
                },
                'uncertainty': {
                    '1d': float,
                    '5d': float,
                    '20d': float
                },
                'volatility': {
                    '1d': float,
                    '5d': float,
                    '20d': float
                },
                'risk_score': float,
                'raw_predictions': dict  # Full model outputs
            }
        """
        # Tokenize OHLC data
        tokens = self.tokenizer.tokenize_ohlc(ohlc_data)

        # Use last 256 tokens for inference
        seq_len = min(256, len(tokens))
        input_tokens = tokens[-seq_len:]

        # Convert to tensor
        token_tensor = torch.LongTensor(input_tokens).unsqueeze(0).to(self.device)

        # Prepare input
        input_data = {'token_ids': token_tensor}

        # Run inference
        with torch.no_grad():
            predictions = self.model(input_data)

        # Extract predictions
        result = self._parse_predictions(predictions, symbol)

        return result

    def _parse_predictions(self, predictions: Dict[str, torch.Tensor], symbol: str) -> Dict:
        """Parse model predictions into standardized format"""

        # Check if continuous horizons are available
        if self.use_continuous_horizons and 'price_reg_continuous' in predictions:
            # Continuous horizon predictions (Chapter 2 architecture)
            continuous_preds = predictions['price_reg_continuous'][0].cpu().numpy()  # [100]
            continuous_unc = predictions['price_uncertainty_continuous'][0].cpu().numpy()  # [100]

            # Find optimal horizon (day with max predicted return)
            optimal_idx = int(np.argmax(continuous_preds))
            optimal_horizon = optimal_idx + 1  # 1-indexed
            optimal_return = float(continuous_preds[optimal_idx])

            # Extract discrete predictions for backward compatibility
            pred_1d = float(continuous_preds[0])
            pred_5d = float(continuous_preds[4])
            pred_20d = float(continuous_preds[19])

            uncertainty_1d = float(continuous_unc[0])
            uncertainty_5d = float(continuous_unc[4])
            uncertainty_20d = float(continuous_unc[19])

            # Determine direction based on optimal horizon prediction
            direction = 'LONG' if optimal_return > 0 else 'SHORT'

            # Confidence based on optimal prediction
            confidence = min(abs(optimal_return) / (continuous_unc[optimal_idx] + 0.01) * 20, 100)

        else:
            # Legacy discrete predictions
            pred_1d = predictions['price_reg_1d'][0, 0].item()
            pred_5d = predictions['price_reg_5d'][0, 0].item()
            pred_20d = predictions['price_reg_20d'][0, 0].item()

            uncertainty_1d = predictions['price_uncertainty_1d'][0, 0].item()
            uncertainty_5d = predictions['price_uncertainty_5d'][0, 0].item()
            uncertainty_20d = predictions['price_uncertainty_20d'][0, 0].item()

            # No continuous data
            continuous_preds = None
            optimal_horizon = 5  # Default to 5-day
            optimal_return = pred_5d

            direction = 'LONG' if pred_5d > 0 else 'SHORT'
            confidence = min(abs(pred_5d) / (uncertainty_5d + 0.01) * 20, 100)

        # Extract volatility predictions (always discrete for now)
        vol_1d = predictions.get('volatility_1d', torch.tensor([[0.02]]))[0, 0].item()
        vol_5d = predictions.get('volatility_5d', torch.tensor([[0.04]]))[0, 0].item()
        vol_20d = predictions.get('volatility_20d', torch.tensor([[0.08]]))[0, 0].item()

        # Extract risk score
        risk_score = predictions.get('risk_score', torch.tensor([[0.5]]))[0, 0].item()

        result = {
            'symbol': symbol,
            'direction': direction,
            'confidence': float(confidence),
            'optimal_horizon': int(optimal_horizon),
            'optimal_return': float(optimal_return),
            'predictions': {
                '1d': float(pred_1d),
                '5d': float(pred_5d),
                '20d': float(pred_20d)
            },
            'uncertainty': {
                '1d': float(uncertainty_1d),
                '5d': float(uncertainty_5d),
                '20d': float(uncertainty_20d)
            },
            'volatility': {
                '1d': float(vol_1d),
                '5d': float(vol_5d),
                '20d': float(vol_20d)
            },
            'risk_score': float(risk_score),
            'raw_predictions': {
                key: value.cpu().numpy().tolist() if isinstance(value, torch.Tensor) else value
                for key, value in predictions.items()
                if key in ['market_regime', 'volatility_regime', 'risk_factors']
            }
        }

        # Add continuous predictions if available
        if continuous_preds is not None:
            result['predictions_continuous'] = continuous_preds.tolist()

        return result

    def batch_predict(self, ohlc_data_list: List[np.ndarray], symbols: List[str]) -> List[Dict]:
        """
        Run batch inference on multiple symbols

        Args:
            ohlc_data_list: List of OHLC arrays
            symbols: List of symbol names

        Returns:
            List of prediction dictionaries
        """
        results = []

        for ohlc_data, symbol in zip(ohlc_data_list, symbols):
            try:
                result = self.predict(ohlc_data, symbol)
                results.append(result)
            except Exception as e:
                print(f"[PredictionEngine] Error predicting {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })

        return results


# Global engine instance (loaded once at server startup)
_engine_instance: Optional[PredictionEngine] = None


def get_prediction_engine(checkpoint_path: Optional[str] = None, device: str = 'cuda') -> PredictionEngine:
    """
    Get or create prediction engine (singleton)

    Args:
        checkpoint_path: Path to checkpoint (only used on first call)
        device: Device to use ('cpu', 'cuda', 'cuda:0') - defaults to cuda

    Returns:
        PredictionEngine instance
    """
    global _engine_instance

    if _engine_instance is None:
        if checkpoint_path is None:
            # Default checkpoint path - use verified continuous horizons model (10 epochs)
            checkpoint_path = str(
                Path(__file__).parent.parent.parent / "models" / "finvec_continuous_10epoch.pt"
            )

        _engine_instance = PredictionEngine(checkpoint_path, device)

    return _engine_instance


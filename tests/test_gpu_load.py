import sys
import os
os.environ['PYTHONPATH'] = '/home/rford/caelum/caelum-supersystem/finvec'
sys.path.insert(0, '/home/rford/caelum/caelum-supersystem/finvec')

from fincoll.inference.prediction_engine import PredictionEngine

print('Loading model on GPU...')
engine = PredictionEngine(
    checkpoint_path='models/finvec_continuous_10epoch.pt',
    device='cuda'
)
print(f'✅ Model loaded! use_continuous_horizons: {engine.use_continuous_horizons}')
print(f'✅ Model parameters: {sum(p.numel() for p in engine.model.parameters()) / 1e6:.1f}M')
